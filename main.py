from tqdm import tqdm
import torch
from torch import optim
import os
import csv
import time
import numpy as np
from eval import eval_model
from argument import get_args
from models import get_net_optimizer_scheduler
from methods import get_model
from datasets import get_dataloaders
from utils.density import get_density, is_gmm_committee



def get_inputs_labels(data):
    if isinstance(data, list):
        inputs = [x.to(args.device) for x in data]
        labels = torch.arange(len(inputs), device=args.device)
        labels = labels.repeat_interleave(inputs[0].size(0))
        inputs = torch.cat(inputs, dim=0)
    else:
        inputs = data.to(args.device)
        labels = torch.zeros(inputs.size(0), device=args.device).long()
    return inputs, labels


def append_timing_row(args, epoch, phase, seconds, task=None):
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "timings.csv")
    row = {
        "epoch": epoch,
        "phase": phase,
        "seconds": seconds,
        "task": task,
        "model_name": args.model.name,
        "method": args.model.method,
        "eval_classifier": args.eval.eval_classifier,
        "data_order": getattr(args, "data_order", None),
        "seed": getattr(args, "seed", None),
    }
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)
    if args.save_path == "./checkpoints":
        args.save_path = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(args.save_path, exist_ok=True)
    if args.model.name in ("dino_v2", "anomaly_dino"):
        layer_idx = getattr(args, "dino_layer_idx", -1)
        layer_list = getattr(args, "dino_layer_indices", "")
        if layer_list:
            layer_label = layer_list
        else:
            layer_label = "last" if layer_idx is None or layer_idx < 0 else str(layer_idx + 1)
        print("=" * 80)
        print(f"[{args.model.name}] Running with intermediate layer: {layer_label}")
        print("=" * 80)
    print(
        f"Running model={args.model.name}, method={args.model.method}, "
        f"eval={args.eval.eval_classifier}, density={args.density.name}"
    )
    net, optimizer, scheduler = get_net_optimizer_scheduler(args)
    density = get_density(args)
    net.to(args.device)

    model = get_model(args, net, optimizer, scheduler)

    dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames = [], [], [], []
    task_wise_mean, task_wise_cov, task_wise_train_data_nums = [], [], []
    for t in range(args.dataset.n_tasks):
        print('---' * 10, f'Task:{t}', '---' * 10)
        train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums, all_test_filenames = get_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames)
        task_wise_train_data_nums.append(data_train_nums)

        extra_para = None
        if args.model.method == 'panda':
            extra_para = model.get_center(train_dataloader)

        net.train()
        last_epoch_embeds = None
        for epoch in tqdm(range(args.train.num_epochs)):
            epoch_start = time.perf_counter()
            one_epoch_embeds = []
            if args.model.method == 'upper':
                for dataloader_train in dataloaders_train:
                    for batch_idx, (data) in enumerate(dataloader_train):
                        inputs, labels = get_inputs_labels(data)
                        model(epoch, inputs, labels, one_epoch_embeds, t, extra_para)
            else:
                for batch_idx, (data) in enumerate(train_dataloader):
                    inputs, labels = get_inputs_labels(data)
                    model(epoch, inputs, labels, one_epoch_embeds, t, extra_para)
            append_timing_row(args, epoch, "train_epoch", time.perf_counter() - epoch_start, task=t)
            last_epoch_embeds = one_epoch_embeds

            if args.train.test_epochs > 0 and (epoch+1) % args.train.test_epochs == 0:
                if not is_gmm_committee(density):
                    eval_start = time.perf_counter()
                    net.eval()
                    density = model.training_epoch(
                        density,
                        one_epoch_embeds,
                        task_wise_mean,
                        task_wise_cov,
                        task_wise_train_data_nums,
                        t,
                    )
                    eval_model(
                        args,
                        epoch,
                        dataloaders_test,
                        learned_tasks,
                        net,
                        density,
                        round_task=t,
                        all_test_filenames=all_test_filenames,
                    )
                    net.train()
                    append_timing_row(args, epoch, "eval", time.perf_counter() - eval_start, task=t)
                else:
                    eval_start = time.perf_counter()
                    net.eval()
                    density = model.training_epoch_gmm(
                        density,
                        one_epoch_embeds,
                        task_wise_mean,
                        task_wise_cov,
                        task_wise_train_data_nums,
                        t,
                        save=False,
                    )
                    eval_model(
                        args,
                        epoch,
                        dataloaders_test,
                        learned_tasks,
                        net,
                        density,
                        round_task=t,
                        all_test_filenames=all_test_filenames,
                    )
                    net.train()
                    append_timing_row(args, epoch, "eval", time.perf_counter() - eval_start, task=t)

        if hasattr(model, 'end_task'):
            model.end_task(train_dataloader)
        if is_gmm_committee(density) and last_epoch_embeds is not None:
            eval_start = time.perf_counter()
            net.eval()
            density = model.training_epoch_gmm(
                density,
                last_epoch_embeds,
                task_wise_mean,
                task_wise_cov,
                task_wise_train_data_nums,
                t,
                save=True,
            )
            eval_model(
                args,
                args.train.num_epochs - 1,
                dataloaders_test,
                learned_tasks,
                net,
                density,
                round_task=t,
                all_test_filenames=all_test_filenames,
            )
            net.train()
            append_timing_row(args, args.train.num_epochs - 1, "eval", time.perf_counter() - eval_start, task=t)

    if args.save_checkpoint:
        torch.save(net,  f'{args.save_path}/net.pth')
        torch.save(density, f'{args.save_path}/density.pth')

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args = get_args()
    main(args)
