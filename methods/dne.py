import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils.base_method import BaseMethod
from utils.dino_layers import parse_dino_layer_indices



class DNE(BaseMethod):
    def __init__(self, args, net, optimizer, scheduler):
        super(DNE, self).__init__(args, net, optimizer, scheduler)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.ewc_fisher = None
        self.ewc_prev_params = None


    def forward(self, 
                epoch, 
                inputs, 
                labels, 
                one_epoch_embeds, 
                t, 
                *args):
        if self.args.dataset.strong_augmentation:
            half_num = int(len(inputs) / 2)
            no_strongaug_inputs = inputs[:half_num]
        else:
            no_strongaug_inputs = inputs

        if (
            self.args.model.name != 'dino_v2'
            and self.args.model.fix_head
            and self.args.ewc_lambda <= 0
        ):
            if t >= 1:
                for param in self.net.head.parameters():
                    param.requires_grad = False

        if self.args.model.name == 'dino_v2':
            with torch.no_grad():
                layer_indices = parse_dino_layer_indices(self.args)
                noaug_embeds = self.net(
                    no_strongaug_inputs,
                    layer_idx=self.args.dino_layer_idx,
                    layer_indices=layer_indices,
                )
                one_epoch_embeds.append(noaug_embeds.cpu())
            return

        self.optimizer.zero_grad()
        with torch.no_grad():
            noaug_embeds = self.net.forward_features(no_strongaug_inputs) # z = f_e(x)
            one_epoch_embeds.append(noaug_embeds.cpu())
        out, _ = self.net(inputs)  # y = h(z)
        loss = self.cross_entropy(out, labels)
        loss = loss + self._ewc_loss()
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step(epoch)

    def _get_inputs_labels(self, data):
        if isinstance(data, list):
            inputs = [x.to(self.args.device) for x in data]
            labels = torch.arange(len(inputs), device=self.args.device)
            labels = labels.repeat_interleave(inputs[0].size(0))
            inputs = torch.cat(inputs, dim=0)
        else:
            inputs = data.to(self.args.device)
            labels = torch.zeros(inputs.size(0), device=self.args.device).long()
        return inputs, labels


    def training_epoch(self, 
                       density, 
                       one_epoch_embeds, 
                       task_wise_mean, 
                       task_wise_cov, 
                       task_wise_train_data_nums, 
                       t):
        if self.args.eval.eval_classifier == 'density':
            one_epoch_embeds = torch.cat(one_epoch_embeds)
            if one_epoch_embeds.dim() == 3:
                one_epoch_embeds = one_epoch_embeds.reshape(-1, one_epoch_embeds.size(-1))
            one_epoch_embeds = F.normalize(one_epoch_embeds, p=2, dim=1)
            mean, cov = density.fit(one_epoch_embeds)

            if len(task_wise_mean) < t + 1:
                task_wise_mean.append(mean)
                task_wise_cov.append(cov)
            else:
                task_wise_mean[-1] = mean
                task_wise_cov[-1] = cov

            task_wise_embeds = []
            for i in range(t + 1):
                if i < t:
                    past_mean = task_wise_mean[i]
                    past_cov = task_wise_cov[i]
                    past_nums =  task_wise_train_data_nums[i]
                    past_embeds = np.random.multivariate_normal(
                        past_mean, 
                        past_cov, 
                        size=int(past_nums * (1 - self.args.noise_ratio))
                    )
                    task_wise_embeds.append(torch.FloatTensor(past_embeds))
                    noise_mean, noise_cov = np.random.rand(past_mean.shape[0]), np.random.rand(past_cov.shape[0], past_cov.shape[1])
                    noise = np.random.multivariate_normal(
                        noise_mean, 
                        noise_cov, 
                        size=int(past_nums * self.args.noise_ratio)
                    )
                    task_wise_embeds.append(torch.FloatTensor(noise))
                else:
                    task_wise_embeds.append(one_epoch_embeds)
            for_eval_embeds = torch.cat(task_wise_embeds, dim=0)
            for_eval_embeds = F.normalize(for_eval_embeds, p=2, dim=1)
            _, _ = density.fit(for_eval_embeds)
            return density
        else:
            pass

    def training_epoch_gmm(
        self,
        density,
        one_epoch_embeds,
        task_wise_mean,
        task_wise_cov,
        task_wise_train_data_nums,
        t,
        save=True,
    ):
        one_epoch_embeds = torch.cat(one_epoch_embeds)
        if one_epoch_embeds.dim() == 3:
            one_epoch_embeds = one_epoch_embeds.reshape(-1, one_epoch_embeds.size(-1))
        one_epoch_embeds = F.normalize(one_epoch_embeds, p=2, dim=1)
        density.fit_task(one_epoch_embeds, task_id=t, save=save)
        return density

    def _ewc_loss(self):
        if (
            self.args.model.name != "vit"
            or self.args.ewc_lambda <= 0
            or self.ewc_fisher is None
            or self.ewc_prev_params is None
        ):
            return 0.0
        penalty = 0.0
        for name, param in self.net.head.named_parameters():
            if not param.requires_grad:
                continue
            fisher = self.ewc_fisher.get(name)
            prev = self.ewc_prev_params.get(name)
            if fisher is None or prev is None:
                continue
            penalty = penalty + (fisher * (param - prev).pow(2)).sum()
        return self.args.ewc_lambda * penalty

    def end_task(self, train_dataloader):
        if self.args.model.name != "vit" or self.args.ewc_lambda <= 0:
            return
        self.net.eval()
        fisher = {name: torch.zeros_like(param) for name, param in self.net.head.named_parameters()}
        batch_count = 0
        for batch_idx, (data) in enumerate(train_dataloader):
            inputs, labels = self._get_inputs_labels(data)
            self.net.zero_grad()
            out, _ = self.net(inputs)
            loss = self.cross_entropy(out, labels)
            loss.backward()
            for name, param in self.net.head.named_parameters():
                if param.grad is None:
                    continue
                fisher[name] += param.grad.detach().pow(2)
            batch_count += 1
            if self.args.ewc_fisher_batches > 0 and batch_count >= self.args.ewc_fisher_batches:
                break
        if batch_count > 0:
            for name in fisher:
                fisher[name] /= float(batch_count)
        if self.ewc_fisher is None:
            self.ewc_fisher = fisher
        else:
            for name in fisher:
                self.ewc_fisher[name] = self.ewc_fisher[name] + fisher[name]
        self.ewc_prev_params = {
            name: param.detach().clone()
            for name, param in self.net.head.named_parameters()
        }
        self.net.train()
