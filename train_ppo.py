"""Train one PPO policy per task with stable-baselines3.

Usage:
    python train_ppo.py                      # all 3 tasks, 200k steps each
    python train_ppo.py --task off_peak_control --timesteps 10000   # smoke run
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from gym_env import GymTrafficEnv
from server.traffic_env import TASK_CONFIGS


def train(task_id: str, timesteps: int) -> None:
    vec_env = make_vec_env(lambda: GymTrafficEnv(task_id), n_envs=8)
    # Load-bearing: raw obs/rewards span 0→200+, PPO needs them normalized.
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,  # tiny action space — avoid premature collapse
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        tensorboard_log="./tb/",
    )
    model.learn(total_timesteps=timesteps, tb_log_name=task_id)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/ppo_{task_id}")
    vec_env.save(f"models/vecnorm_{task_id}.pkl")  # required for correct eval
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASK_CONFIGS), help="default: all tasks")
    parser.add_argument("--timesteps", type=int, default=200_000)
    args = parser.parse_args()

    for task_id in [args.task] if args.task else list(TASK_CONFIGS):
        print(f"=== training {task_id} ({args.timesteps} steps) ===")
        train(task_id, args.timesteps)
