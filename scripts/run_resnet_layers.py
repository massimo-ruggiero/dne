#!/usr/bin/env python3
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="./configs/cad.yaml")
    parser.add_argument("--results_dir", default="./results_resnet_layers")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--layer_start", type=int, default=1)
    parser.add_argument("--layer_end", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    args, extra = parser.parse_known_args()

    base_results = args.results_dir.rstrip("/\\")
    for layer in range(args.layer_start, args.layer_end + 1):
        results_dir = f"{base_results}_layer{layer}"
        cmd = [
            args.python,
            "main.py",
            "--config-file",
            args.config_file,
            "--data_dir",
            args.data_dir,
            "--results_dir",
            results_dir,
            "--resnet_layer_idx",
            str(layer),
            "--resnet_freeze_backbone",
            "true",
            *extra,
        ]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
