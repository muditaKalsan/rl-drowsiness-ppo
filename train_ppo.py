"""
Train PPO (or A2C) on the DrowsinessEnv.
Supports:
  --quick      -> short 20k timestep run
  --algo {ppo,a2c}
  --sweep      -> train PPO for several WAIT_PENALTY values (0,-0.02,-0.05,-0.1,-0.2)
    The best model for each penalty is saved under `models/` with a suffix.
"""

import argparse
import os
import numpy as np
import joblib
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from config import (
    MODELS_DIR,
    RESULTS_DIR,
    PPO_PARAMS,
    QUICK_TIMESTEPS,
    MAX_TIMESTEPS,
    RANDOM_SEED,
    WAIT_PENALTY,
)
from drowsiness_env import DrowsinessEnv

def train_one(penalty, total_timesteps, algo="ppo"):
    # Create environments with the appropriate wait penalty
    train_env = lambda: Monitor(DrowsinessEnv(mode="train", wait_penalty=penalty))
    eval_env = lambda: Monitor(DrowsinessEnv(mode="val", wait_penalty=penalty))
    
    # callbacks
    eval_cb = EvalCallback(
        eval_env(),
        best_model_save_path=MODELS_DIR,
        log_path=RESULTS_DIR,
        eval_freq=2000,
        deterministic=True,
        render=False,
        n_eval_episodes=200,
    )
    ckpt_cb = CheckpointCallback(save_freq=5000, save_path=MODELS_DIR,
                                 name_prefix=f"ckpt_pen{penalty}")

    if algo == "ppo":
        model = PPO(**PPO_PARAMS, env=train_env())
    else:
        a2c_params = PPO_PARAMS.copy()
        a2c_params.pop("net_arch", None)
        model = A2C(**a2c_params, env=train_env())

    print(f"[INFO] Training {algo.upper()} with WAIT_PENALTY={penalty} for {total_timesteps:,} steps ...")
    model.learn(total_timesteps=total_timesteps, callback=[eval_cb, ckpt_cb])
    # save final model with penalty suffix
    model_path = MODELS_DIR / f"{algo}_pen{penalty}_final.zip"
    model.save(str(model_path))
    print(f"[INFO] Saved final model to {model_path}")
    return model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="short 20k step run")
    parser.add_argument("--algo", choices=["ppo", "a2c"], default="ppo", help="algorithm")
    parser.add_argument("--sweep", action="store_true", help="train PPO for several WAIT_PENALTY values")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_timesteps = QUICK_TIMESTEPS if args.quick else MAX_TIMESTEPS

    if args.sweep:
        penalties = [0.0, -0.02, -0.05, -0.1, -0.2]
        for p in penalties:
            train_one(p, total_timesteps, algo=args.algo)
    else:
        # use the penalty from config (default -0.05)
        train_one(WAIT_PENALTY, total_timesteps, algo=args.algo)

if __name__ == "__main__":
    main()
