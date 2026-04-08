import numpy as np
import random
import matplotlib.pyplot as plt

class BangaloreTrafficEnv:
    def __init__(self):
        self.queues = [0, 0, 0, 0]
        self.green_phase = 0
        self.time_step = 0
        self.max_steps = 100
        self.rush_hour = True

    def reset(self):
        lam = 8 if self.rush_hour else 3
        self.queues = [np.random.poisson(lam) for _ in range(4)]
        self.green_phase = 0
        self.time_step = 0
        return self.state()

    def step(self, action):
        self.green_phase = action
        if action == 0:
            self.queues[0] = max(0, self.queues[0] - 3)
            self.queues[1] = max(0, self.queues[1] - 3)
        else:
            self.queues[2] = max(0, self.queues[2] - 3)
            self.queues[3] = max(0, self.queues[3] - 3)
        lam = 8 if self.rush_hour else 3
        arrivals = [np.random.poisson(lam // 3) for _ in range(4)]
        self.queues = [q + a for q, a in zip(self.queues, arrivals)]
        reward = -sum(self.queues)
        self.time_step += 1
        done = self.time_step >= self.max_steps
        return self.state(), reward, done, {}

    def state(self):
        return np.array(self.queues + [self.green_phase, self.time_step])