"""
Orchestrate the entire RL drowsiness‑detection pipeline.
Running this script executes the steps in the correct order:
    1. data_utils.py   -> cache features (scaled)
    2. baseline.py     -> train & evaluate SVM / Random Forest
    3. train_ppo.py    -> PPO (or A2C) training (default PPO)
    4. evaluate.py     -> evaluate PPO (or A2C) + baselines on test set
    5. make_plots.py   -> generate all required figures
    6. export_excel.py -> tiny Excel overview
Use the flag ``--quick`` for a short 20 k‑step PPO run (smoke test).
"""

import argparse
import subprocess
import sys
from pathlib import Path

def _run_cmd(cmd, description):
    print(f"\n=== {description} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[ERROR] {description} failed (exit code {result.returncode}).")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run PPO with a short 20k‑step schedule")
    args = parser.parse_args()

    py = sys.executable

    # 1. Feature extraction & scaling
    _run_cmd([py, "data_utils.py"], "Feature extraction & scaling")

    # 2. Baselines (train & evaluate)
    _run_cmd([py, "baseline.py"], "Baseline classifiers")

    # 3. PPO training
    ppo_cmd = [py, "train_ppo.py"]
    if args.quick:
        ppo_cmd.append("--quick")
    _run_cmd(ppo_cmd, "PPO training (default PPO)")

    # 4. Evaluation (RL + baselines)
    _run_cmd([py, "evaluate.py"], "Evaluation of PPO and baselines")

    # 5. Plot generation
    _run_cmd([py, "make_plots.py"], "Plot generation")

    # 6. Excel export
    _run_cmd([py, "export_excel.py"], "Excel overview export")

    print("\n=== All steps completed successfully ===")

if __name__ == "__main__":
    main()
