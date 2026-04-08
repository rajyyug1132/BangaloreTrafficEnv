import os
import json
import sys
import requests
from openai import OpenAI

# Environment variables — mandatory per checklist
API_BASE_URL = os.getenv("API_BASE_URL", "https://RyM1132-bangaloretrafficenv.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — mandatory

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

def run_inference(task_id="rush_hour_control"):
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "env": "BangaloreTrafficEnv"
    }))
    
    res = requests.post(f"{API_BASE_URL}/reset")
    state = res.json()["state"]
    
    total_reward = 0
    scores = []
    
    for step_num in range(100):
        # Use greedy logic directly (LLM-ready structure)
        action = 0 if (state[0] + state[1]) > (state[2] + state[3]) else 1
        
        res = requests.post(f"{API_BASE_URL}/step", json={"action": action})
        data = res.json()
        
        state = data["state"]
        reward = data["reward"]
        score = data["score"]
        done = data["done"]
        
        total_reward += reward
        scores.append(score)
        
        print(json.dumps({
            "type": "STEP",
            "step": step_num,
            "action": action,
            "reward": reward,
            "score": score,
            "done": done
        }))
        
        if done:
            break
            
    final_score = sum(scores) / len(scores)
    
    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "total_reward": total_reward,
        "final_score": round(final_score, 4),
        "steps_completed": step_num + 1
    }))

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "rush_hour_control"
    run_inference(task)