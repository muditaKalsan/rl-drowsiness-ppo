"""
Custom Gymnasium environment for the RL drowsiness detection task.
The environment scales observations using the StandardScaler that was fit
on the training subjects (saved in `data/scaler.pkl`).
It also validates itself with `check_env` and runs a quick random‑agent
sanity check when instantiated with `validate=True`.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from config import (
    WAIT_PENALTY,
    CHUNK_LENGTH_SECONDS,
    FS,
    SCALER_PATH,
    RANDOM_SEED,
)
from features import get_splits
import joblib

class DrowsinessEnv(gym.Env):
    """Gymnasium environment for the RL drowsiness detection task.

    Observation space (float32 Box):
        - newest_chunk_features (n_features)
        - running_mean_features (n_features)
        - elapsed_fraction (1)
    Action space (Discrete(3)):
        0 = predict alert
        1 = predict drowsy
        2 = wait (reveal next chunk)
    """

    metadata = {"render_modes": []}

    def __init__(self, mode="train", validate=False):
        super().__init__()
        self.mode = mode
        self.rng = np.random.default_rng(RANDOM_SEED)

        # Load data according to mode
        splits = get_splits()
        if mode == "train":
            self.X, self.y = splits["train"]
        elif mode == "val":
            self.X, self.y = splits["val"]
        elif mode == "test":
            self.X, self.y = splits["test"]
        else:
            raise ValueError("mode must be 'train', 'val' or 'test'")

        self.n_samples, self.n_chunks, self.n_features = self.X.shape

        # Load scaler and store for scaling observations
        self.scaler = joblib.load(SCALER_PATH)

        low = np.full(self.n_features * 2 + 1, -np.inf, dtype=np.float32)
        high = np.full(self.n_features * 2 + 1,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # internal state placeholders
        self.current_idx = None
        self.current_chunk = None
        self.cum_sum = None
        self.done = False

        if validate:
            # sanity‑check the environment
            from stable_baselines3.common.env_checker import check_env
            print("[INFO] Running env_checker …")
            check_env(self, warn=True)
            # run a short random‑policy test (10 episodes)
            print("[INFO] Running quick random‑policy test …")
            for _ in range(10):
                obs, _ = self.reset()
                done = False
                while not done:
                    action = self.action_space.sample()
                    obs, reward, terminated, truncated, info = self.step(action)
                    done = terminated or truncated
            print("[INFO] Random‑policy test completed.")

    # ------------------------------------------------------------------
    def _reset_state(self, idx):
        self.current_idx = idx
        self.current_chunk = 0
        self.cum_sum = np.zeros(self.n_features, dtype=np.float32)
        self.done = False

    def _current_obs(self):
        newest = self.X[self.current_idx, self.current_chunk]
        # scale the raw features
        newest = self.scaler.transform(newest.reshape(1, -1)).ravel()
        self.cum_sum += newest
        mean = self.cum_sum / (self.current_chunk + 1)
        elapsed = (self.current_chunk + 1) / self.n_chunks
        return np.concatenate([newest, mean, [elapsed]]).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.mode == "train":
            idx = self.rng.integers(0, self.n_samples)
        else:
            # sequential for deterministic evaluation
            if not hasattr(self, "_eval_counter"):
                self._eval_counter = 0
            idx = self._eval_counter % self.n_samples
            self._eval_counter += 1
        self._reset_state(idx)
        return self._current_obs(), {"sample_index": idx}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        reward = 0.0
        terminated = False
        info = {}
        true_label = int(self.y[self.current_idx])

        if action == 2:  # WAIT
            reward = WAIT_PENALTY
            if self.current_chunk == self.n_chunks - 1:
                # timeout
                terminated = True
                reward = -1.0
                info.update({"prediction": None, "correct": False,
                             "delay": self.n_chunks * CHUNK_LENGTH_SECONDS})
            else:
                self.current_chunk += 1
                return self._current_obs(), reward, terminated, False, info
        else:  # make prediction
            pred = action  # 0 or 1
            correct = pred == true_label
            reward = 1.0 if correct else -1.0
            terminated = True
            delay = (self.current_chunk + 1) * CHUNK_LENGTH_SECONDS
            info.update({"prediction": pred, "correct": correct, "delay": delay})
            return self._current_obs(), reward, terminated, False, info

        # fallback (should not reach)
        return self._current_obs(), reward, terminated, False, info

    def render(self):
        raise NotImplementedError
