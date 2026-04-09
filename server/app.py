from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from server.traffic_env import BangaloreTrafficEnv, TASK_CONFIGS

app = FastAPI(title="BangaloreTrafficEnv", version="1.0")
env = BangaloreTrafficEnv()

# Grader callable paths — must match openenv.yaml and graders.py exactly
TASK_GRADERS = {
    "rush_hour_control": "graders:grade_rush_hour",
    "off_peak_control":  "graders:grade_off_peak",
    "sustained_flow":    "graders:grade_sustained_flow",
}

# ── Data models ──────────────────────────────────────────
class ResetRequest(BaseModel):
    task: Optional[str] = None

class StepRequest(BaseModel):
    action: int   # 0 = NS green, 1 = EW green

class StepResponse(BaseModel):
    state: list
    reward: float
    done: bool
    info: dict
    score: float  # Normalised strictly in (0, 1)

class StateResponse(BaseModel):
    state: list

class ResetResponse(BaseModel):
    state: list
    task: str

# ── Health / metadata / schema (required by platform validator) ──
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "BangaloreTrafficEnv",
        "description": (
            "A real-world RL environment simulating a Bangalore traffic intersection. "
            "An AI agent learns to dynamically control traffic signals to minimize "
            "congestion across North, South, East and West lanes."
        ),
        "version": "1.0",
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "discrete",
            "n": 2,
            "description": "0 = North-South green, 1 = East-West green",
        },
        "observation": {
            "type": "array",
            "shape": [6],
            "description": "[cars_north, cars_south, cars_east, cars_west, green_phase, time_step]",
        },
        "state": {
            "type": "array",
            "shape": [6],
            "description": "Same as observation",
        },
    }

# ── Core endpoints ───────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "env": "BangaloreTrafficEnv"}

@app.get("/tasks")
def list_tasks():
    """List all available tasks — grader field must NOT be 'default'."""
    return {
        "tasks": [
            {
                "id": task_id,
                "name": cfg.get("description", task_id),
                "difficulty": cfg.get("difficulty", "medium"),
                "grader": TASK_GRADERS[task_id],   # real callable path, never "default"
            }
            for task_id, cfg in TASK_CONFIGS.items()
        ]
    }

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()):
    """Reset the environment. Optionally pass a task id to switch scenarios."""
    state = env.reset(task_id=req.task)
    return {"state": state.tolist(), "task": env.task_id}

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    state, reward, done, info = env.step(req.action)
    score = env.compute_score(reward)
    return {
        "state": state.tolist(),
        "reward": reward,
        "done": done,
        "info": info,
        "score": score,
    }

@app.get("/state", response_model=StateResponse)
def get_state():
    return {"state": env.state().tolist()}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()