from sklearn.metrics import roc_curve, auc, roc_auc_score
import csv
import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from argument import get_args
from datasets import get_dataloaders
from utils.visualization import plot_tsne, compare_histogram, cal_anomaly_map
from utils.patch_masking import select_foreground_patches
import cv2
from utils.dino_layers import parse_dino_layer_indices



def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def append_metric_row(args, epoch, task_id, task_name, roc_auc, round_task=None):
    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics.csv")
    row = {
        "epoch": epoch,
        "model_name": args.model.name,
        "density": args.density.name,
        "round_task": round_task,
        "task_id": task_id,
        "task_name": task_name,
        "auc": roc_auc,
    }
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _unnormalize_image(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_tensor.dtype, device=img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img_tensor.dtype, device=img_tensor.device)
    img = img_tensor * std[:, None, None] + mean[:, None, None]
    return img.clamp(0, 1)


def _save_heatmap(img_tensor, heatmap, save_path, alpha=0.5):
    img = _unnormalize_image(img_tensor).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_MAGMA)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)

def csflow_eval(args, epoch, dataloaders_test, learned_tasks, net, round_task=None):
    all_roc_auc = []
    eval_task_wise_scores, eval_task_wise_labels = [], []
    task_num = 0
    for idx, (dataloader_test, learned_task) in enumerate(zip(dataloaders_test, learned_tasks)):
        test_z, test_labels = list(), list()

        with torch.no_grad():
            for i, data in enumerate(dataloader_test):
                inputs, labels = data
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                _, z, jac = net(inputs)
                z = t2np(z[..., None])
                score = np.mean(z ** 2, axis=(1, 2))
                test_z.append(score)
                test_labels.append(t2np(labels))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        anomaly_score = np.concatenate(test_z, axis=0)
        roc_auc = roc_auc_score(is_anomaly, anomaly_score)
        all_roc_auc.append(roc_auc * len(learned_task))
        task_num += len(learned_task)
        print('data_type:', learned_task, 'auc:', roc_auc, '**' * 11)
        append_metric_row(args, epoch, idx, ",".join(learned_task), roc_auc, round_task=round_task)
        append_metric_row(args, epoch, ",".join(learned_task), roc_auc)

        eval_task_wise_scores.append(anomaly_score)
        eval_task_wise_scores_np = np.concatenate(eval_task_wise_scores)
        eval_task_wise_labels.append(is_anomaly)
        eval_task_wise_labels_np = np.concatenate(eval_task_wise_labels)
    mean_auc = np.sum(all_roc_auc) / task_num
    print('mean_auc:', mean_auc, '**' * 11)
    append_metric_row(args, epoch, -1, "mean", mean_auc, round_task=round_task)

    if args.eval.visualization:
        task_dir = os.path.join(args.results_dir, f"task_{round_task}" if round_task is not None else "task_eval")
        name = f'{args.model.method}_task{len(learned_tasks)}_epoch{epoch}'
        his_save_path = os.path.join(
            task_dir,
            f'his_results/{args.model.method}{args.model.name}_{args.train.num_epochs}_epochs_seed{args.seed}',
        )
        compare_histogram(np.array(eval_task_wise_scores_np), np.array(eval_task_wise_labels_np), start=0, thresh=5,
                          interval=1, name=name, save_path=his_save_path)


def revdis_eval(args, epoch, dataloaders_test, learned_tasks, net, round_task=None):
    all_roc_auc = []
    eval_task_wise_scores, eval_task_wise_labels = [], []
    task_num = 0
    for idx, (dataloader_test, learned_task) in enumerate(zip(dataloaders_test, learned_tasks)):
        gt_list_sp, pr_list_sp = [], []
        with torch.no_grad():
            for img, gt, label in dataloader_test:
                img = img.to(args.device)
                inputs = net.encoder(img)
                outputs = net.decoder(net.bn(inputs))
                anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
                pr_list_sp.append(np.max(anomaly_map))

        roc_auc = roc_auc_score(gt_list_sp, pr_list_sp)
        all_roc_auc.append(roc_auc * len(learned_task))
        task_num += len(learned_task)
        print('data_type:', learned_task, 'auc:', roc_auc, '**' * 11)
        append_metric_row(args, epoch, idx, ",".join(learned_task), roc_auc, round_task=round_task)
        append_metric_row(args, epoch, ",".join(learned_task), roc_auc)

        eval_task_wise_scores.append(pr_list_sp)
        eval_task_wise_scores_np = np.concatenate(eval_task_wise_scores)
        eval_task_wise_labels.append(gt_list_sp)
        eval_task_wise_labels_np = np.concatenate(eval_task_wise_labels)
    mean_auc = np.sum(all_roc_auc) / task_num
    print('mean_auc:', mean_auc, '**' * 11)
    append_metric_row(args, epoch, -1, "mean", mean_auc, round_task=round_task)

    if args.eval.visualization:
        task_dir = os.path.join(args.results_dir, f"task_{round_task}" if round_task is not None else "task_eval")
        name = f'{args.model.method}_task{len(learned_tasks)}_epoch{epoch}'
        his_save_path = os.path.join(
            task_dir,
            f'his_results/{args.model.method}{args.model.name}_{args.train.num_epochs}_epochs_seed{args.seed}',
        )
        compare_histogram(np.array(eval_task_wise_scores_np), np.array(eval_task_wise_labels_np), thresh=2, interval=1,
                          name=name, save_path=his_save_path)



def eval_model(args, epoch, dataloaders_test, learned_tasks, net, density, round_task=None, all_test_filenames=None):
    if args.model.method == 'csflow':
        csflow_eval(args, epoch, dataloaders_test, learned_tasks, net, round_task=round_task)
    elif args.model.method == 'revdis':
        revdis_eval(args, epoch, dataloaders_test, learned_tasks, net, round_task=round_task)
    else:
        all_roc_auc, all_embeds, all_labels = [], [], []
        task_num = 0
        for idx, (dataloader_test,  learned_task) in enumerate(zip(dataloaders_test, learned_tasks)):
            labels, embeds, logits = [], [], []
            imgs = []
            filenames = None
            if all_test_filenames is not None and idx < len(all_test_filenames):
                filenames = all_test_filenames[idx]
            with torch.no_grad():
                for x, label in dataloader_test:
                    if args.model.name in ('dino_v2', 'anomaly_dino'):
                        use_patch_tokens = args.model.name == 'anomaly_dino'
                        layer_indices = parse_dino_layer_indices(args) if args.model.name == 'dino_v2' else None
                        embed = net(
                            x.to(args.device),
                            layer_idx=args.dino_layer_idx,
                            patch_tokens=use_patch_tokens,
                            layer_indices=layer_indices,
                        )
                        embeds.append(embed.cpu())
                    else:
                        logit, embed = net(x.to(args.device))
                        _, logit = torch.max(logit, 1)
                        logits.append(logit.cpu())
                        embeds.append(embed.cpu())
                    labels.append(label.cpu())
                    if args.model.name == 'anomaly_dino':
                        imgs.append(x.cpu())
            labels, embeds = torch.cat(labels), torch.cat(embeds)
            if logits:
                logits = torch.cat(logits)
            if imgs:
                imgs = torch.cat(imgs)
            # norm embeds
            if args.eval.eval_classifier == 'density':
                if args.model.name == 'anomaly_dino' and embeds.dim() == 3:
                    image_size = args.dataset.image_size
                    patch_size = getattr(net.backbone, "patch_size", None)
                    distances = []
                    heatmaps = []
                    for i in range(embeds.size(0)):
                        patches = embeds[i].numpy()
                        selected, mask = select_foreground_patches(
                            patches,
                            image_size=image_size,
                            patch_size=patch_size,
                            threshold=args.dino_mask_threshold,
                            kernel_size=args.dino_mask_kernel,
                            border=args.dino_mask_border,
                            min_center_ratio=args.dino_mask_min_center_ratio,
                        )
                        selected = torch.from_numpy(selected)
                        selected = F.normalize(selected, p=2, dim=1)
                        scores = density.predict(selected)
                        scores = torch.as_tensor(scores)
                        k = args.dino_patch_top_k
                        if k <= 0 or k > scores.numel():
                            k = scores.numel()
                        topk_scores, _ = torch.topk(scores, k=k, dim=0)
                        distances.append(torch.mean(topk_scores).item())
                        full_scores = np.full((mask.shape[0],), float(scores.min().item()), dtype=np.float32)
                        full_scores[mask] = scores.cpu().numpy()
                        grid = int(np.sqrt(full_scores.shape[0]))
                        heatmap = full_scores.reshape(grid, grid)
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        heatmap = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                        heatmaps.append(heatmap)
                    distances = np.array(distances)
                else:
                    embeds = F.normalize(embeds, p=2, dim=1)  # embeds.shape=(2*bs, emd_dim)
                    distances = density.predict(embeds)  # distances.shape=(2*bs)
                fpr, tpr, _ = roc_curve(labels, distances)
            elif args.eval.eval_classifier == 'head':
                if args.model.name == 'dino_v2':
                    raise ValueError("eval_classifier='head' not supported for dino_v2 without a head.")
                fpr, tpr, _ = roc_curve(labels, logits)
            roc_auc = auc(fpr, tpr)
            all_roc_auc.append(roc_auc * len(learned_task))
            task_num += len(learned_task)
            all_embeds.append(embeds)
            all_labels.append(labels)
            print('data_type:', learned_task[:], 'auc:', roc_auc, '**' * 11)
            append_metric_row(args, epoch, idx, ",".join(learned_task), roc_auc, round_task=round_task)

            if args.model.name == 'anomaly_dino' and args.eval.eval_classifier == 'density':
                task_dir = os.path.join(
                    args.results_dir,
                    f"anomaly_maps/task_{round_task}" if round_task is not None else "anomaly_maps/task_eval",
                    f"component_{'_'.join(learned_task)}",
                )
                for i in range(labels.size(0)):
                    if filenames is not None and i < len(filenames):
                        filename = str(filenames[i])
                        base = os.path.basename(filename)
                        defect = os.path.basename(os.path.dirname(filename))
                        class_name = os.path.basename(os.path.dirname(os.path.dirname(filename)))
                        name = f"{class_name}_{defect}_{base}"
                    else:
                        name = f"img_{i:06d}.png"
                    save_path = os.path.join(task_dir, name)
                    _save_heatmap(imgs[i], heatmaps[i], save_path, alpha=0.5)

            if args.eval.visualization and args.model.name != 'anomaly_dino':
                task_dir = os.path.join(args.results_dir, f"task_{round_task}" if round_task is not None else "task_eval")
                name = f'{args.model.method}_task{len(learned_tasks)}_{learned_task[0]}_epoch{epoch}'
                data_order = getattr(args, "data_order", None)
                if data_order is None and hasattr(args, "dataset"):
                    data_order = getattr(args.dataset, "dataset_order", None)
                his_save_path = os.path.join(
                    task_dir,
                    f'his_results/{args.model.method}{args.model.name}_{args.train.num_epochs}e_order{data_order}_seed{args.seed}',
                )
                tnse_save_path = os.path.join(
                    task_dir,
                    f'tsne_results/{args.model.method}{args.model.name}_{args.train.num_epochs}e_order{data_order}_seed{args.seed}',
                )
                plot_tsne(labels, np.array(embeds), defect_name=name, save_path=tnse_save_path)
                # These parameters can be modified based on the visualization effect
                start, thresh, interval = 0, 120, 1
                compare_histogram(np.array(distances), labels, start=start,
                                  thresh=thresh, interval=interval,
                                  name=name, save_path=his_save_path)

        mean_auc = np.sum(all_roc_auc) / task_num
        print('mean_auc:', mean_auc, '**' * 11)
        append_metric_row(args, epoch, -1, "mean", mean_auc, round_task=round_task)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)
    if args.save_path == "./checkpoints":
        args.save_path = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(args.save_path, exist_ok=True)
    dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames = [], [], [], []
    for t in range(args.dataset.n_tasks):
        train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums, all_test_filenames = get_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames)

    epoch = args.train.num_epochs
    net, density = torch.load(f'{args.save_path}/net.pth'), torch.load(f'{args.save_path}/density.pth')
    eval_model(args, epoch, dataloaders_test, learned_tasks, net, density, round_task=None)
