"""
Evaluate PPO (or A2C) models and the baseline classifiers on the test subjects.
The script:
1. Loads the best PPO model for each WAIT_PENALTY (or the single default).
2. Runs the policy *deterministically* on every test sample once.
3. Records predictions, correctness, detection delay and the action taken at each step.
4. Computes accuracy, macro‑F1, confusion matrix, average delay, timeout rate,
   and per‑subject accuracy.
5. Saves a CSV/Excel summary and also stores the action‑distribution matrix
   (shape: n_penalties x n_chunks) for later plotting.
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from config import (
    MODELS_DIR,
    RESULTS_DIR,
    SCALER_PATH,
    WAIT_PENALTY,
    CHUNK_LENGTH_SECONDS,
)
from drowsiness_env import DrowsinessEnv
from baseline import run_baselines

def _load_model(penalty, algo="ppo"):
    # The best model saved by EvalCallback has the name "best_model.zip"
    # when doing a sweep we saved models with suffix _pen{penalty}_final.zip; the best one is also saved with that suffix.
    # For simplicity we load the *final* model for each penalty.
    model_path = MODELS_DIR / f"{algo}_pen{penalty}_final.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    env = DrowsinessEnv(mode="test")
    if algo == "ppo":
        return PPO.load(str(model_path), env=env)
    else:
        return A2C.load(str(model_path), env=env)

def evaluate_model(model, env, penalty):
    X_test, y_test, subjects = get_splits()["test"]
    n_samples = X_test.shape[0]
    preds = []
    delays = []
    # action counts per chunk index (0‑based). We count only "wait" actions; the final action is either alert/drowsy.
    action_counts = np.zeros(env.n_chunks, dtype=int)
    for i in range(n_samples):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if action == 2:  # wait
                action_counts[env.current_chunk] += 1
            if terminated:
                preds.append(info.get("prediction"))
                delays.append(info.get("delay"))
                break
    preds = np.array(preds, dtype=int)
    delays = np.array(delays, dtype=float)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds)
    timeout_rate = np.mean(preds == None)  # should be zero for deterministic policy
    avg_delay = delays.mean()
    per_subject = {}
    for subj in np.unique(subjects):
        mask = subjects == subj
        per_subject[int(subj)] = accuracy_score(y_test[mask], preds[mask])
    return {
        "penalty": penalty,
        "accuracy": acc,
        "macro_f1": f1,
        "confusion": cm,
        "avg_delay": avg_delay,
        "timeout_rate": timeout_rate,
        "per_subject": per_subject,
        "action_counts": action_counts,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="ppo", help="algorithm name (ppo or a2c)")
    parser.add_argument("--sweep", action="store_true", help="evaluate all penalties used during sweep")
    args = parser.parse_args()

    # Baselines – they also save confusion matrices to results/ already
    print("[INFO] Running baselines …")
    baseline_metrics = run_baselines()

    # RL evaluation
    print("[INFO] Evaluating RL models …")
    penalties = [WAIT_PENALTY]
    if args.sweep:
        penalties = [0.0, 0.02, 0.05, 0.1, 0.2]
    rl_results = []
    for p in penalties:
        model = _load_model(p, algo=args.algo)
        env = DrowsinessEnv(mode="test")
        # sanity‑check env
        check_env(env, warn=True)
        rl_res = evaluate_model(model, env, p)
        rl_results.append(rl_res)

    # Save CSV/Excel
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in rl_results:
        rows.append({
            "method": f"PPO_pen{r['penalty']}",
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "avg_delay_s": r["avg_delay"],
            "timeout_rate": r["timeout_rate"],
        })
    for name, m in baseline_metrics.items():
        rows.append({
            "method": name,
            "accuracy": m["accuracy"],
            "macro_f1": m["f1"],
            "avg_delay_s": np.nan,
            "timeout_rate": np.nan,
        })
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Metrics CSV saved to {csv_path}")

    # Save action counts for each penalty (for plotting)
    for r in rl_results:
        np.save(RESULTS_DIR / f"action_counts_pen{r['penalty']}.npy", r["action_counts"])

    # Save Excel workbook with extra sheets
    xl_path = RESULTS_DIR / "results.xlsx"
    with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="summary", index=False)
        # per‑subject accuracies for each penalty
        for r in rl_results:
            subj_df = pd.DataFrame(list(r["per_subject"].items()), columns=["subject", "accuracy"])
            sheet_name = f"ppo_pen{r['penalty']}_per_subj"
            subj_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"[INFO] Excel workbook saved to {xl_path}")

if __name__ == "__main__":
    main()
