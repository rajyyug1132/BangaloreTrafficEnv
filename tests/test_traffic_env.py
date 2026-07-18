"""Tests for the core simulator: dynamics, scoring bounds, task configs."""

import numpy as np
import pytest

from server.traffic_env import BangaloreTrafficEnv, TASK_CONFIGS, DEFAULT_TASK


def test_all_tasks_have_required_keys():
    for task_id, cfg in TASK_CONFIGS.items():
        assert {"lam", "arrival_lam", "max_steps", "difficulty", "description"} <= set(cfg)


def test_reset_returns_6dim_state():
    env = BangaloreTrafficEnv()
    state = env.reset()
    assert state.shape == (6,)
    assert state.dtype == np.float32
    assert state[4] == 0  # green phase
    assert state[5] == 0  # time step


def test_reset_switches_task():
    env = BangaloreTrafficEnv("rush_hour_control")
    env.reset(task_id="off_peak_control")
    assert env.task_id == "off_peak_control"
    assert env.arrival_lam == TASK_CONFIGS["off_peak_control"]["arrival_lam"]


def test_unknown_task_falls_back_to_default():
    env = BangaloreTrafficEnv("no_such_task")
    assert env.task_id == DEFAULT_TASK
    env.reset(task_id="also_bogus")  # must not raise or switch
    assert env.task_id == DEFAULT_TASK


def test_step_discharges_green_axis_only():
    env = BangaloreTrafficEnv("off_peak_control")
    env.reset()
    env.queues = [10, 10, 10, 10]
    env.arrival_lam = 0  # silence arrivals for determinism
    env.step(0)
    assert env.queues[0] == 7 and env.queues[1] == 7   # NS discharged by 3
    assert env.queues[2] == 10 and env.queues[3] == 10  # EW untouched


def test_queues_never_negative():
    env = BangaloreTrafficEnv("off_peak_control")
    env.reset()
    env.queues = [1, 0, 0, 0]
    env.arrival_lam = 0
    env.step(0)
    assert all(q >= 0 for q in env.queues)


def test_reward_is_negative_total_queue():
    env = BangaloreTrafficEnv("off_peak_control")
    env.reset()
    _, reward, _, _ = env.step(0)
    assert reward == -sum(env.queues)


def test_episode_ends_at_max_steps():
    env = BangaloreTrafficEnv("off_peak_control")
    env.reset()
    done = False
    for i in range(env.max_steps):
        _, _, done, _ = env.step(i % 2)
    assert done


def test_sustained_flow_reports_success_in_final_info():
    np.random.seed(0)
    env = BangaloreTrafficEnv("sustained_flow")
    env.reset()
    info = {}
    for _ in range(env.max_steps):
        # greedy control keeps queues low under arrival_lam=1
        action = 0 if env.queues[0] + env.queues[1] >= env.queues[2] + env.queues[3] else 1
        _, _, done, info = env.step(action)
    assert "avg_queue_per_lane" in info
    assert info["sustained_flow_success"] is True


@pytest.mark.parametrize("task_id", list(TASK_CONFIGS))
def test_compute_score_strictly_in_open_interval(task_id):
    np.random.seed(1)
    env = BangaloreTrafficEnv(task_id)
    env.reset()
    assert env.compute_score(0.0) >= 0.001  # before any step
    for _ in range(10):
        _, reward, _, _ = env.step(1)
    score = env.compute_score(reward)
    assert 0.001 <= score <= 0.999
