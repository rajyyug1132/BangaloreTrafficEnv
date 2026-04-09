"""
Grader functions for BangaloreTrafficEnv.
Each function receives the episode history (list of step dicts) and returns
a normalised score strictly in (0.0, 1.0) — 0.0 and 1.0 are not accepted.

Expected step dict format (produced by /step endpoint):
    {
        "state":  [...],
        "reward": float,   # negative sum of waiting cars
        "done":   bool,
        "score":  float,   # per-step normalised score from server
        "info":   dict
    }
"""

from typing import List, Dict, Any

# Worst-case reward per step: 4 lanes × ~20 cars = -80
_MAX_PENALTY_PER_STEP = 80.0


def _reward_to_score(rewards: List[float]) -> float:
    """Shift reward range [-MAX_PENALTY, 0] → score strictly in (0.0, 1.0)."""
    if not rewards:
        return 0.001
    n = len(rewards)
    worst = -_MAX_PENALTY_PER_STEP * n
    total = sum(rewards)
    if worst == 0:
        return 0.999
    score = (total - worst) / (0.0 - worst)
    return float(min(max(score, 0.001), 0.999))  # strictly in (0, 1)


def grade_rush_hour(episode: List[Dict[str, Any]]) -> float:
    """
    Rush Hour Traffic (Hard) — Poisson lambda=8.
    Score = normalised cumulative reward over 100 steps.
    """
    rewards = [step.get("reward", 0.0) for step in episode]
    return _reward_to_score(rewards)


def grade_off_peak(episode: List[Dict[str, Any]]) -> float:
    """
    Off Peak Traffic (Easy) — Poisson lambda=3.
    Score = normalised cumulative reward over 100 steps.
    """
    rewards = [step.get("reward", 0.0) for step in episode]
    return _reward_to_score(rewards)


def grade_sustained_flow(episode: List[Dict[str, Any]]) -> float:
    """
    Sustained Flow (Medium) — Keep average queue < 5 cars per lane.
    Score = 1.0 if constraint satisfied the whole episode, degrades linearly.
    """
    if not episode:
        return 0.001

    total_queue = 0.0
    steps = 0
    for step in episode:
        state = step.get("state", [])
        if len(state) >= 4:
            total_queue += sum(state[:4])   # North + South + East + West
            steps += 1

    if steps == 0:
        return 0.001

    avg_queue_per_lane = total_queue / (steps * 4)
    threshold = 5.0
    # Full score when avg < threshold; degrades linearly above
    score = 1.0 - max(0.0, avg_queue_per_lane - threshold) / threshold
    return float(min(max(score, 0.001), 0.999))  # strictly in (0, 1)
