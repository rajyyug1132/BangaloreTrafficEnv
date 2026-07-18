"""Compare PPO vs fixed-timer and greedy baselines: 50 seeded episodes per task.

Usage:
    python eval.py               # all controllers (needs trained models)
    python eval.py --skip-ppo    # baselines only
"""

import argparse
import csv
from statistics import mean

from baselines import FixedTimer, Greedy
from gym_env import GymTrafficEnv
from server.traffic_env import TASK_CONFIGS

N_EPISODES = 50
FIXED_KS = [1, 2, 3, 5, 10]
QUEUE_THRESHOLD = 5.0


class PPOController:
    def __init__(self, task_id: str):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        dummy = DummyVecEnv([lambda: GymTrafficEnv(task_id)])
        self.vecnorm = VecNormalize.load(f"models/vecnorm_{task_id}.pkl", dummy)
        self.vecnorm.training = False
        self.model = PPO.load(f"models/ppo_{task_id}")

    def reset(self):
        pass

    def act(self, state) -> int:
        obs = self.vecnorm.normalize_obs(state)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def run_episode(env: GymTrafficEnv, policy, seed: int):
    policy.reset()
    obs, _ = env.reset(seed=seed)
    total_reward, queue_sum, steps = 0.0, 0.0, 0
    info, done = {}, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
        done = terminated or truncated
        total_reward += reward
        queue_sum += float(sum(obs[:4]))
        steps += 1
    avg_queue = queue_sum / (steps * 4)
    success = bool(info.get("sustained_flow_success", avg_queue < QUEUE_THRESHOLD))
    return total_reward, avg_queue, success


def main(skip_ppo: bool) -> None:
    rows = []
    summary = []
    for task_id in TASK_CONFIGS:
        env = GymTrafficEnv(task_id)
        controllers = {}
        if not skip_ppo:
            controllers["ppo"] = PPOController(task_id)
        controllers["greedy"] = Greedy()
        for k in FIXED_KS:
            controllers[f"fixed_{k}"] = FixedTimer(k)

        for name, policy in controllers.items():
            episodes = [run_episode(env, policy, seed) for seed in range(N_EPISODES)]
            for seed, (r, q, s) in enumerate(episodes):
                rows.append([task_id, name, seed, f"{r:.2f}", f"{q:.3f}", s])
            summary.append(
                (
                    task_id,
                    name,
                    mean(e[0] for e in episodes),
                    mean(e[1] for e in episodes),
                    sum(e[2] for e in episodes) / len(episodes),
                )
            )

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["task", "controller", "seed", "total_reward", "avg_queue_per_lane", "success"]
        )
        writer.writerows(rows)

    print(f"\n{'task':<20} {'controller':<10} {'mean_reward':>12} {'avg_queue':>10} {'success':>8}")
    for task_id, name, r, q, s in summary:
        print(f"{task_id:<20} {name:<10} {r:>12.1f} {q:>10.2f} {s:>7.0%}")
    print(f"\nWrote {len(rows)} rows to results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ppo", action="store_true", help="baselines only")
    args = parser.parse_args()
    main(skip_ppo=args.skip_ppo)
