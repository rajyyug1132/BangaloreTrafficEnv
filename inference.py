import os
import requests
from openai import OpenAI

# 1. YOUR HUGGING FACE ENVIRONMENT
# Hardcoded so it does not conflict with the Hackathon's Proxy URL
ENV_URL = "https://rym1132-bangaloretrafficenv.hf.space"

# 2. STRICT COMPLIANCE: INITIALIZE OPENAI CLIENT EXACTLY AS REQUESTED
# The validator requires os.environ for these specific keys to prove you are using their proxy.
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

def get_llm_action(state):
    """Ask the LLM proxy to decide the action based on traffic state."""
    prompt = f"""
    You are an intelligent traffic light controller.
    The current traffic queue lengths are: {state}.
    Action 0: Green for North-South.
    Action 1: Green for East-West.
    If North-South queues (index 0,1) are longer, output 0. Otherwise output 1.
    Respond with ONLY a single integer (0 or 1).
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        return int(response.choices[0].message.content.strip())
    except Exception as e:
        # Loud error so if the proxy rejects us, we can read exactly WHY in the validator log
        print(f"\n[CRITICAL LLM ERROR] The proxy rejected the call: {str(e)}\n")
        return 0 if state[0] + state[1] > state[2] + state[3] else 1

def run_inference(task_id="rush_hour_control"):
    print(f"[START] task_id={task_id} total_steps=100")
    
    try:
        # Reset with network safety
        res = requests.post(f"{ENV_URL}/reset", timeout=10)
        res.raise_for_status() 
        data = res.json()
        state = data.get("state", [0, 0, 0, 0, 0, 0])
        
        total_reward = 0
        success = True
        step_num = 0

        for step_num in range(100):
            # ACTUAL API CALL HAPPENS HERE
            action = get_llm_action(state)
            
            # Step the environment with network safety
            try:
                res = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=10)
                res.raise_for_status()
                data = res.json()
                
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
        # Full failure handling
        print(f"[END] success=false steps=0 score=0.0000 rewards=0.0 error={str(e)}")

if __name__ == "__main__":
    run_inference(task_id="rush_hour_control")