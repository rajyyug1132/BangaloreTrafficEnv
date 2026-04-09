from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from server.traffic_env import BangaloreTrafficEnv, TASK_CONFIGS

app = FastAPI(title="BangaloreTrafficEnv")
env = BangaloreTrafficEnv()

# ── Data models ──────────────────────────────────────────
class ResetRequest(BaseModel):
    task: Optional[str] = None   # e.g. "rush_hour_control", "off_peak_control", "sustained_flow"

class StepRequest(BaseModel):
    action: int                  # 0 = NS green, 1 = EW green

class StepResponse(BaseModel):
    state: list
    reward: float
    done: bool
    info: dict
    score: float                 # Normalised 0.0–1.0

class StateResponse(BaseModel):
    state: list

class ResetResponse(BaseModel):
    state: list
    task: str

# ── Endpoints ────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "env": "BangaloreTrafficEnv"}

@app.get("/tasks")
def list_tasks():
    """List all available tasks with their descriptions."""
    return {
        "tasks": [
            {
                "id": task_id,
                "name": cfg.get("description", task_id),
                "grader": "default"
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