import os
import sys
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://rym1132-bangaloretrafficenv.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

def run_inference(task_id="rush_hour_control"):
    # STRICT [START] format
    print(f"[START] task_id={task_id} total_steps=100")
    
    # Robust reset call
res = requests.post(f"{API_BASE_URL}/reset")
data = res.json()

# Check if 'state' actually exists to avoid the KeyError
if "state" in data:
    state = data["state"]
else:
    # Fallback to a default state if the server fails
    print(f"[DEBUG] Server error: {data}")
    state = [0, 0, 0, 0, 0, 0]
    
    total_reward = 0
    scores = []
    raw_rewards = []
    
    for step_num in range(100):
        # Baseline greedy heuristic
        action = 0 if (state[0] + state[1]) > (state[2] + state[3]) else 1
        
        res = requests.post(f"{API_BASE_URL}/step", json={"action": action})
        data = res.json()
        
       def run_inference(task_id="rush_hour_control"):
    print(f"[START] task_id={task_id} total_steps=100")
    
    try:
        # Reset with safety
        res = requests.post(f"{API_BASE_URL}/reset", timeout=10)
        res.raise_for_status() # Check for 404/500 errors
        data = res.json()
        state = data.get("state", [0, 0, 0, 0, 0, 0])
        
        total_reward = 0
        success = True

        for step_num in range(100):
            # Your simple greedy logic
            action = 0 if state[0] + state[1] > state[2] + state[3] else 1
            
            # Step with safety WRAPPER
            try:
                res = requests.post(f"{API_BASE_URL}/step", json={"action": action}, timeout=10)
                res.raise_for_status()
                data = res.json()
                
                # Use .get() to prevent KeyError 'state'
                state = data.get("state", state) 
                reward = data.get("reward", 0.0)
                done = data.get("done", False)
                
                total_reward += reward
                print(f"[STEP] step={step_num} action={action} reward={reward:.2f} done={str(done).lower()} error=null")
                
                if done:
                    break
            except Exception as e:
                print(f"[STEP] step={step_num} action={action} reward=0.0 done=true error={str(e)}")
                success = False
                break

        # Final Score Calculation
        final_score = max(0.0, min(1.0, (total_reward + 5000) / 5000))
        print(f"[END] success={str(success).lower()} steps={step_num + 1} score={final_score:.4f} rewards={total_reward:.2f}")

    except Exception as e:
        # This handles a total failure if the server is unreachable
        print(f"[END] success=false steps=0 score=0.0000 rewards=0.0 error={str(e)}")