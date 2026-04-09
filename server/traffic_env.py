import numpy as np

# Task configurations
TASK_CONFIGS = {
    "rush_hour_control": {
        "lam": 8,
        "max_steps": 100,
        "description": "Peak traffic burst: Poisson lambda=8",
    },
    "off_peak_control": {
        "lam": 3,
        "max_steps": 100,
        "description": "Standard traffic flow: Poisson lambda=3",
    },
    "sustained_flow": {
        "lam": 5,
        "max_steps": 100,
        "queue_threshold": 5,
        "description": "Maintain average queue < 5 per lane over 100 steps",
    },
}

DEFAULT_TASK = "rush_hour_control"


class BangaloreTrafficEnv:
    def __init__(self, task_id: str = DEFAULT_TASK):
        self.task_id = task_id if task_id in TASK_CONFIGS else DEFAULT_TASK
        self.config = TASK_CONFIGS[self.task_id]
        self.queues = [0, 0, 0, 0]
        self.green_phase = 0
        self.time_step = 0
        self.max_steps = self.config["max_steps"]
        self.lam = self.config["lam"]
        # For sustained_flow: track cumulative average queue
        self._total_queue_sum = 0
        self._steps_counted = 0

    def reset(self, task_id: str = None):
        if task_id and task_id in TASK_CONFIGS:
            self.task_id = task_id
            self.config = TASK_CONFIGS[self.task_id]
            self.max_steps = self.config["max_steps"]
            self.lam = self.config["lam"]

        self.queues = [np.random.poisson(self.lam) for _ in range(4)]
        self.green_phase = 0
        self.time_step = 0
        self._total_queue_sum = 0
        self._steps_counted = 0
        return self.state()

    def step(self, action: int):
        self.green_phase = action
        # Discharge green-phase lanes
        if action == 0:  # NS green
            self.queues[0] = max(0, self.queues[0] - 3)
            self.queues[1] = max(0, self.queues[1] - 3)
        else:            # EW green
            self.queues[2] = max(0, self.queues[2] - 3)
            self.queues[3] = max(0, self.queues[3] - 3)

        # New arrivals
        arrivals = [np.random.poisson(max(1, self.lam // 3)) for _ in range(4)]
        self.queues = [q + a for q, a in zip(self.queues, arrivals)]

        reward = -sum(self.queues)

        # Track for sustained_flow grading
        self._total_queue_sum += sum(self.queues)
        self._steps_counted += 1

        self.time_step += 1
        done = self.time_step >= self.max_steps

        info = {
            "task_id": self.task_id,
            "time_step": self.time_step,
        }

        if done and self.task_id == "sustained_flow":
            avg_queue = self._total_queue_sum / (self._steps_counted * 4)
            info["avg_queue_per_lane"] = round(avg_queue, 3)
            threshold = self.config.get("queue_threshold", 5)
            info["sustained_flow_success"] = avg_queue < threshold

        return self.state(), reward, done, info

    def compute_score(self, reward: float) -> float:
        """Return normalised score in [0.0, 1.0] for the current task."""
        if self.task_id == "sustained_flow":
            # Score based on whether average queue stays below threshold
            if self._steps_counted == 0:
                return 0.0
            avg_queue = self._total_queue_sum / (self._steps_counted * 4)
            threshold = self.config.get("queue_threshold", 5)
            # Full score if avg < threshold, degrades linearly after
            return float(np.clip(1.0 - max(0, avg_queue - threshold) / threshold, 0.0, 1.0))
        else:
            # Worst case per step: 4 lanes × 20 cars = 80 waiting
            max_penalty = 80
            return float(np.clip((reward + max_penalty) / max_penalty, 0.0, 1.0))

    def state(self):
        return np.array(self.queues + [self.green_phase, self.time_step], dtype=np.float32)