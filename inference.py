import os
import sys
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://RyM1132-bangaloretrafficenv.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

def run_inference(task_id="rush_hour_control"):
    # STRICT [START] format
    print(f"[START] task={task_id} env=BangaloreTrafficEnv model={MODEL_NAME}")
    
    res = requests.post(f"{API_BASE_URL}/reset")
    state = res.json()["state"]
    
    total_reward = 0
    scores = []
    raw_rewards = []
    
    for step_num in range(100):
        # Baseline greedy heuristic
        action = 0 if (state[0] + state[1]) > (state[2] + state[3]) else 1
        
        res = requests.post(f"{API_BASE_URL}/step", json={"action": action})
        data = res.json()
        
        state = data["state"]
        reward = data["reward"]
        score = data["score"]
        done = data["done"]
        
        total_reward += reward
        scores.append(score)
        raw_rewards.append(reward)
        
        # STRICT [STEP] format
        print(f"[END] success={str(success).lower()} steps={step_num + 1} score={final_score:.4f} rewards={rewards_str}")   
        
        if done:
            break
            
    final_score = sum(scores) / len(scores)
    success = final_score > 0.8
    rewards_str = ",".join([f"{r:.2f}" for r in raw_rewards])
    
    # STRICT [END] format
    print(f"[END] success={str(success).lower()} steps={step_num + 1} score={final_score:.4f} rewards={rewards_str}")

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "rush_hour_control"
    run_inference(task)