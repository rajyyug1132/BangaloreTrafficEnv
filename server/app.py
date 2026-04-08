from fastapi import FastAPI
from pydantic import BaseModel
from server.traffic_env import BangaloreTrafficEnv
import numpy as np

app = FastAPI(title="BangaloreTrafficEnv")
env = BangaloreTrafficEnv()

# ── Data models ──────────────────────────────────────────
class StepRequest(BaseModel):
    action: int          # 0 = NS green, 1 = EW green

class StepResponse(BaseModel):
    state: list
    reward: float
    done: bool
    info: dict
    score: float         # Normalised 0.0–1.0 for grader

class StateResponse(BaseModel):
    state: list

# ── Endpoints ────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "env": "BangaloreTrafficEnv"}

@app.post("/reset", response_model=StateResponse)
def reset():
    state = env.reset()
    return {"state": state.tolist()}

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    state, reward, done, info = env.step(req.action)

    # Normalise reward to 0.0–1.0
    # Worst case: 4 lanes × 20 cars × 100 steps = -8000 per step total
    max_penalty = 8000
    score = float(np.clip((reward + max_penalty) / max_penalty, 0.0, 1.0))

    return {
        "state": state.tolist(),
        "reward": reward,
        "done": done,
        "info": info,
        "score": score
    }

@app.get("/state", response_model=StateResponse)
def get_state():
    return {"state": env.state().tolist()}
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()