"""Core simulator for a four-lane Bangalore traffic intersection.

Dynamics per step:
  1. The green axis's two lanes each discharge up to ``DISCHARGE_PER_LANE`` cars.
  2. Every lane receives Poisson(``arrival_lam``) new arrivals.
  3. Reward is ``-sum(all four queues)`` — zero is the (unreachable) optimum.

Episodes are ``max_steps`` long. Queues are unbounded: when arrivals outpace
the maximum discharge (rush_hour_control, deliberately), congestion grows
regardless of policy.
"""

from typing import Optional

import numpy as np

# Worst-case reward per step assumed by score normalisation: 4 lanes × ~20 cars.
# An assumption, not a bound — scores are clamped, not the queues.
MAX_PENALTY_PER_STEP = 80.0

# Cars discharged per green lane per step (2 lanes per axis → 6 cars/step max).
DISCHARGE_PER_LANE = 3

TASK_CONFIGS: dict = {
    "rush_hour_control": {
        "lam": 8,            # initial queue seed, Poisson per lane
        "arrival_lam": 3,    # per-step arrivals, Poisson per lane (saturated by design)
        "max_steps": 100,
        "difficulty": "hard",
        "description": "Peak traffic burst: Poisson lambda=8",
    },
    "off_peak_control": {
        "lam": 3,
        "arrival_lam": 1,
        "max_steps": 100,
        "difficulty": "easy",
        "description": "Standard traffic flow: Poisson lambda=3",
    },
    "sustained_flow": {
        "lam": 5,
        # arrival_lam=2 made the task unwinnable (8 arrivals/step vs 6 max
        # discharge); 1 keeps avg-queue<5 achievable, per task description
        "arrival_lam": 1,
        "max_steps": 100,
        "difficulty": "medium",
        "queue_threshold": 5,
        "description": "Maintain average queue < 5 per lane over 100 steps",
    },
}

DEFAULT_TASK = "rush_hour_control"


class BangaloreTrafficEnv:
    """Single-intersection queueing simulator with task-switchable dynamics.

    State vector (float32, shape ``(6,)``):
    ``[cars_north, cars_south, cars_east, cars_west, green_phase, time_step]``.

    Uses the global numpy RNG (seed with ``np.random.seed`` for reproducibility).
    """

    def __init__(self, task_id: str = DEFAULT_TASK):
        self.task_id = task_id if task_id in TASK_CONFIGS else DEFAULT_TASK
        self.queues = [0, 0, 0, 0]
        self.green_phase = 0
        self.time_step = 0
        self._apply_config()
        self._reset_episode_stats()

    def _apply_config(self) -> None:
        self.config = TASK_CONFIGS[self.task_id]
        self.max_steps = self.config["max_steps"]
        self.lam = self.config["lam"]
        self.arrival_lam = self.config["arrival_lam"]

    def _reset_episode_stats(self) -> None:
        self._total_queue_sum = 0
        self._total_reward = 0.0
        self._steps_counted = 0

    def reset(self, task_id: Optional[str] = None) -> np.ndarray:
        """Start a new episode; optionally switch task (unknown ids are ignored)."""
        if task_id and task_id in TASK_CONFIGS:
            self.task_id = task_id
            self._apply_config()

        self.queues = [np.random.poisson(self.lam) for _ in range(4)]
        self.green_phase = 0
        self.time_step = 0
        self._reset_episode_stats()
        return self.state()

    def step(self, action: int):
        """Advance one step. ``action``: 0 = NS green, 1 = EW green.

        Returns ``(state, reward, done, info)``.
        """
        self.green_phase = action
        green_lanes = (0, 1) if action == 0 else (2, 3)
        for lane in green_lanes:
            self.queues[lane] = max(0, self.queues[lane] - DISCHARGE_PER_LANE)

        arrivals = [np.random.poisson(self.arrival_lam) for _ in range(4)]
        self.queues = [q + a for q, a in zip(self.queues, arrivals)]

        reward = -sum(self.queues)

        self._total_queue_sum += sum(self.queues)
        self._total_reward += reward
        self._steps_counted += 1

        self.time_step += 1
        done = self.time_step >= self.max_steps

        info = {"task_id": self.task_id, "time_step": self.time_step}
        if done and self.task_id == "sustained_flow":
            avg_queue = self._avg_queue_per_lane()
            threshold = self.config.get("queue_threshold", 5)
            info["avg_queue_per_lane"] = round(avg_queue, 3)
            info["sustained_flow_success"] = avg_queue < threshold

        return self.state(), reward, done, info

    def _avg_queue_per_lane(self) -> float:
        return self._total_queue_sum / (self._steps_counted * 4)

    def compute_score(self, reward: float) -> float:
        """Normalised episode score strictly in (0.0, 1.0), consistent with graders.py.

        sustained_flow: full score at/below the queue threshold, smooth 1/x
        decay above. Other tasks: cumulative reward shifted against the
        ``MAX_PENALTY_PER_STEP`` worst case.
        """
        if self._steps_counted == 0:
            return 0.001
        if self.task_id == "sustained_flow":
            threshold = self.config.get("queue_threshold", 5)
            score = threshold / max(threshold, self._avg_queue_per_lane())
        else:
            worst = -MAX_PENALTY_PER_STEP * self._steps_counted
            score = (self._total_reward - worst) / (0.0 - worst)
        return float(np.clip(score, 0.001, 0.999))

    def state(self) -> np.ndarray:
        """Current observation vector."""
        return np.array(self.queues + [self.green_phase, self.time_step], dtype=np.float32)
