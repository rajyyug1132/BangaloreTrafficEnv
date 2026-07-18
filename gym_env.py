"""Gymnasium wrapper around BangaloreTrafficEnv — in-process, no HTTP."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from server.traffic_env import BangaloreTrafficEnv, TASK_CONFIGS


class GymTrafficEnv(gym.Env):
    def __init__(self, task_id: str = "rush_hour_control"):
        assert task_id in TASK_CONFIGS, f"unknown task: {task_id}"
        self.task_id = task_id
        self.inner = BangaloreTrafficEnv(task_id)
        self.action_space = spaces.Discrete(2)
        # Queues are unbounded (rush_hour arrivals outpace discharge) — no finite high.
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)  # inner env uses the global numpy RNG
        return self.inner.reset(self.task_id), {}

    def step(self, action):
        obs, reward, done, info = self.inner.step(int(action))
        # 100-step cutoff is a time limit → truncation, not termination
        return obs, float(reward), False, done, info
