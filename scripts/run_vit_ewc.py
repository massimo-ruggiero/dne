#!/usr/bin/env python3
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="./configs/cad.yaml")
    parser.add_argument("--results_dir", default="./results_vit_gmm")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--lambdas", default="10,100,1000")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    args, extra = parser.parse_known_args()

    base_results = args.results_dir.rstrip("/\\")
    lambdas = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    for lam in lambdas:
        results_dir = f"{base_results}_{lam}"
        cmd = [
            args.python,
            "main.py",
            "--config-file",
            args.config_file,
            "--data_dir",
            args.data_dir,
            "--results_dir",
            results_dir,
            "--ewc_lambda",
            lam,
            *extra,
        ]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
