---
title: BangaloreTrafficEnv
colorFrom: green
colorTo: red
sdk: docker
pinned: false
---

# BangaloreTrafficEnv

## Environment Description and Motivation
BangaloreTrafficEnv is a real-world Reinforcement Learning environment simulating a high-volume traffic intersection in Bangalore. Fixed-duration traffic signals cannot dynamically adapt to real-time vehicle flow, causing massive congestion. This environment allows AI agents to learn optimal traffic signal switching dynamically to minimize overall wait times across four lanes. 

## Action Space
**Type:** Discrete (2)
* `0`: Set North-South lanes to Green.
* `1`: Set East-West lanes to Green.

## Observation Space
**Type:** Array (Shape: [6])
* `[0]`: Cars waiting North
* `[1]`: Cars waiting South
* `[2]`: Cars waiting East
* `[3]`: Cars waiting West
* `[4]`: Current Green Phase (0 for NS, 1 for EW)
* `[5]`: Current Time Step

## Tasks
1. **rush_hour_control (Hard):** Minimize congestion during peak traffic bursts (Poisson lambda=8) over 100 steps. 
2. **off_peak_control (Easy):** Minimize congestion during standard traffic flow (Poisson lambda=3) over 100 steps.
3. **sustained_flow (Medium):** Maintain average queue size below 5 cars per lane over 100 steps.

## Setup and Usage Instructions
1. Clone the repository and install dependencies:
   `pip install -r requirements.txt`
2. Start the FastAPI server locally:
   `uvicorn app:app --reload --port 7860`
3. Run the baseline inference script:
   `python inference.py rush_hour_control`

## Baseline Scores
Running the baseline greedy heuristic on `rush_hour_control` yields an approximate normalized score of ~0.99 over 100 steps.