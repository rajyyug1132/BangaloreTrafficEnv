"""Non-learned controllers. Shared interface: reset(), act(state) -> int."""


class FixedTimer:
    """Pretimed signal: switch phase every k steps regardless of queues."""

    def __init__(self, k: int):
        self.k = k
        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, state) -> int:
        action = (self.t // self.k) % 2
        self.t += 1
        return action


class Greedy:
    """Serve the axis with more waiting cars (same as inference._fallback_action)."""

    def reset(self):
        pass

    def act(self, state) -> int:
        return 0 if state[0] + state[1] >= state[2] + state[3] else 1
