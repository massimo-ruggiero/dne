#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import yaml
import timm


def _load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def _get_num_layers(dino_name, img_size):
    model = timm.create_model(
        dino_name,
        pretrained=False,
        num_classes=0,
        img_size=img_size,
    )
    try:
        return len(model.blocks)
    finally:
        del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="./configs/cad.yaml")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--layer_start", type=int, default=2)
    parser.add_argument("--layer_step", type=int, default=2)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    args, extra = parser.parse_known_args()

    cfg = _load_config(args.config_file)
    dino_name = cfg.get("model", {}).get("dino_name", "vit_base_patch14_dinov2")
    img_size = cfg.get("dataset", {}).get("image_size", 224)

    num_layers = _get_num_layers(dino_name, img_size)
    layer_end = args.layer_end or num_layers

    if args.layer_start < 1 or args.layer_start > num_layers:
        raise ValueError(f"layer_start must be in [1, {num_layers}]")
    if layer_end < args.layer_start or layer_end > num_layers:
        raise ValueError(f"layer_end must be in [{args.layer_start}, {num_layers}]")
    if args.layer_step <= 0:
        raise ValueError("layer_step must be > 0")

    base_results = args.results_dir.rstrip("/\\")
    layer_numbers = range(args.layer_start, layer_end + 1, args.layer_step)

    for layer_num in layer_numbers:
        layer_idx = layer_num - 1  # convert to 0-based
        results_dir = f"{base_results}_layer{layer_num}"
        cmd = [
            args.python,
            "main.py",
            "--config-file",
            args.config_file,
            "--results_dir",
            results_dir,
            "--dino_layer_idx",
            str(layer_idx),
            *extra,
        ]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
