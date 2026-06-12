#!/usr/bin/env python3
"""Overnight experiment runner — runs continuously, saves progress, survives crashes.

Runs E1 (main comparison), E10 (efficiency), E2 (tau scan) in sequence.
Each experiment auto-saves checkpoints. If power fails, resume with --resume.
"""

import subprocess
import sys
import os
import time

EXPERIMENTS = [
    # (exp_id, extra_args, description, estimated_time_min)
    ("1", "--epochs 50 --seeds 1", "E1: Main Performance Comparison", 40),
    ("10", "--epochs 10 --seeds 1", "E10: Parameter Efficiency", 5),
    ("2", "", "E2: τ-Scan (Singular Value Threshold)", 5),
]

BASE_CMD = [
    sys.executable, "-m", "main.experiments.run_all",
    "--data", "data/synthetic_train.npy",
    "--pretrained", "checkpoints/quickstart/pretrained_generator.pt",
    "--results_dir", "results/overnight",
]

def main():
    os.makedirs("results/overnight", exist_ok=True)
    log_path = "results/overnight/runner.log"

    with open(log_path, "a") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"Overnight runner started at {time.ctime()}\n")
        log.write(f"{'='*60}\n")

    total_start = time.time()
    for exp_id, extra, desc, eta in EXPERIMENTS:
        cmd = BASE_CMD + ["--exp", exp_id] + extra.split()
        print(f"\n{'#'*60}")
        print(f"# {desc} (est. {eta} min)")
        print(f"# {' '.join(cmd)}")
        print(f"{'#'*60}")

        t0 = time.time()
        try:
            result = subprocess.run(cmd, timeout=eta * 120 + 300)  # 2x margin + 5min
            elapsed = (time.time() - t0) / 60
            status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        except subprocess.TimeoutExpired:
            elapsed = (time.time() - t0) / 60
            status = "TIMEOUT"

        with open(log_path, "a") as log:
            log.write(f"[{time.ctime()}] {desc}: {status} ({elapsed:.1f} min)\n")

        print(f"{desc}: {status} ({elapsed:.1f} min)")

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"Overnight runner finished. Total: {total_elapsed:.1f} min")
    print(f"Results: results/overnight/")
    print(f"Log:     results/overnight/runner.log")


if __name__ == "__main__":
    main()
