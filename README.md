---
title: BangaloreTrafficEnv
colorFrom: green
colorTo: red
sdk: docker
pinned: false
---

# BangaloreTrafficEnv

BangaloreTrafficEnv is a Reinforcement Learning environment simulating a high-volume
traffic intersection in Bangalore. Fixed-duration traffic signals cannot adapt to
real-time vehicle flow; this environment lets agents learn dynamic signal switching
to minimize total wait time across four lanes.

## Action Space
**Type:** Discrete (2)
* `0`: Set North-South lanes to Green.
* `1`: Set East-West lanes to Green.

## Observation Space
**Type:** Array (Shape: [6])
* `[0]`–`[3]`: Cars waiting North / South / East / West
* `[4]`: Current Green Phase (0 for NS, 1 for EW)
* `[5]`: Current Time Step

Reward per step is `-sum(all four queues)`. Episodes run 100 steps.

## Tasks
1. **rush_hour_control (Hard):** Minimize congestion during peak traffic bursts (Poisson lambda=8).
2. **off_peak_control (Easy):** Minimize congestion during standard traffic flow (Poisson lambda=3).
3. **sustained_flow (Medium):** Maintain average queue size below 5 cars per lane.

## Results: PPO vs baselines

A PPO agent (stable-baselines3, one policy per task, 200k steps each) compared
against a greedy heuristic (serve the axis with more waiting cars) and pretimed
fixed-timer signals (switch every k steps), over 50 seeded episodes per controller
per task:

| Task | PPO | Greedy | Best fixed-timer |
|---|---|---|---|
| off_peak (mean total reward) | **−689.0** | −691.3 | −776.9 (k=1) |
| sustained_flow (mean total reward / success rate) | −738.0 / **100%** | −736.6 / 100% | −839.1 / 100% (k=1) |
| rush_hour (mean total reward) | −33884.5 | −33564.9 | −33566.9 (k=1) |

![PPO vs baselines: mean avg queue per lane per task](results_comparison.png)

### Interpretation

- **PPO learns the task.** It statistically matches the greedy heuristic on
  off_peak (slightly ahead) and sustained_flow (within noise), and both clearly
  beat every fixed timer. That is the expected ceiling: with two actions and full
  queue observability, greedy is near-optimal for this MDP, so "PPO ≈ greedy ≫
  fixed" is success, not a shortfall.
- **Key metric — sustained_flow success rate: 100% for PPO** (avg queue 1.84 vs
  the 5.0 threshold). Fixed timers only start failing at k=10.
- **sustained_flow was rebalanced** (`arrival_lam` 2 → 1): at 2, arrivals
  (~8 cars/step) exceeded the maximum discharge (6 cars/step), making the task's
  own success criterion (avg queue < 5) impossible for *any* controller — every
  policy scored 0%. The rebalance restores the intended medium difficulty; it is
  a bug fix to match the task description, not result tuning.
- **rush_hour is saturated by design.** Arrivals (~12 cars/step across lanes)
  far exceed the maximum discharge (6 cars/step), so queues grow roughly linearly
  for *every* controller; the <1% spread between PPO, greedy, and fixed_1 is noise
  on an uncontrollable trend, and 0% success across the board is a property of the
  environment, not the agents. It is kept deliberately as a stress scenario.

## Reproduce

```bash
pip install -r requirements.txt stable-baselines3 gymnasium tensorboard matplotlib

# 1. Train one PPO policy per task (writes models/ppo_<task>.zip + VecNormalize stats)
python train_ppo.py                    # all 3 tasks, 200k steps each
python train_ppo.py --task off_peak_control --timesteps 10000   # quick smoke run

# 2. Evaluate PPO vs greedy vs fixed-timer baselines (50 seeded episodes each)
python eval.py                         # writes results.csv + summary table
python eval.py --skip-ppo              # baselines only, no models needed

# 3. Render the comparison chart
python plot_results.py                 # writes results_comparison.png
```

Training and evaluation wrap the simulator in-process via `gym_env.py`
(a `gymnasium.Env` around `server/traffic_env.py`) — no HTTP server needed.

## Server / HTTP agents

```bash
uvicorn server.app:app --reload --port 7860   # start the FastAPI env server
python inference.py rush_hour_control          # LLM-driven agent over HTTP
```
