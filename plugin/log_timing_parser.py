#!/usr/bin/env python3
# Copyright 2025 Google LLC
"""Real-time log parser to extract and upload timing metrics to Wandb"""

import re
import time
import argparse
import wandb
from pathlib import Path


def parse_timing_from_log(log_file: str, wandb_run_id: str, wandb_project: str):
    """
    Parse training log in real-time and upload timing metrics to wandb.

    Extracts:
    - Train loop finished time (rollout + actor update)
    - Actor training step time
    """
    # Initialize wandb run
    run = wandb.init(
        project=wandb_project,
        id=wandb_run_id,
        resume="allow"
    )

    log_path = Path(log_file)
    step = 0

    # Track file position
    with open(log_path, 'r') as f:
        # Go to end of existing content
        f.seek(0, 2)

        print(f"Watching {log_file} for timing metrics...")

        while True:
            line = f.readline()

            if not line:
                time.sleep(0.5)
                continue

            # Parse "Train loop finished in: XX seconds"
            match = re.search(r'Train loop finished in:\s+([\d\.]+)\s+seconds', line)
            if match:
                train_loop_time = float(match.group(1))

                # This is the total time for rollout + reward + actor update
                wandb.log({
                    "timing/train_loop_total_sec": train_loop_time,
                    "_step": step
                })

                print(f"Step {step}: Train loop = {train_loop_time:.4f}s")
                step += 1

            # Parse actor training progress
            match = re.search(r'Actor Training:\s+\d+%.*?(\d+)/(\d+).*?actor_train_loss=([\d\.e\-]+)', line)
            if match:
                current_step = int(match.group(1))
                total_steps = int(match.group(2))

                # Only log when step increases
                if current_step > step:
                    step = current_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse training logs and upload timing to wandb")
    parser.add_argument("--log_file", required=True, help="Path to training log file")
    parser.add_argument("--wandb_run_id", required=True, help="Wandb run ID to resume")
    parser.add_argument("--wandb_project", default="qwen3-grpo-gsm8k", help="Wandb project name")

    args = parser.parse_args()

    try:
        parse_timing_from_log(args.log_file, args.wandb_run_id, args.wandb_project)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        raise
