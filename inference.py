import requests
import json
import sys

BASE_URL = "http://localhost:7860"

def run_inference(task_id="rush_hour_control"):
    # Set rush hour based on task
    rush = task_id != "off_peak_control"

    # START log — mandatory format
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "env": "BangaloreTrafficEnv"
    }))

    # Reset environment
    res = requests.post(f"{BASE_URL}/reset")
    state = res.json()["state"]

    total_reward = 0
    scores = []

    for step_num in range(100):
        # Greedy action: green for busier side
        action = 0 if (state[0] + state[1]) > (state[2] + state[3]) else 1

        res = requests.post(f"{BASE_URL}/step", json={"action": action})
        data = res.json()

        state = data["state"]
        reward = data["reward"]
        score = data["score"]
        done = data["done"]
        total_reward += reward
        scores.append(score)

        # STEP log — mandatory format
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

    # END log — mandatory format
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