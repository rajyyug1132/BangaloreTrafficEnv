"""
Inference script for BangaloreTrafficEnv
=========================================
Follows the mandatory STDOUT format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# ── Environment & model configuration ────────────────────────────────────────
# Mandatory variables (as required by the evaluation platform)
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # optional: used only with from_docker_image()

# Derived helpers
ENV_URL  = os.getenv("ENV_URL", "https://rym1132-bangaloretrafficenv.hf.space")
API_KEY  = HF_TOKEN or os.getenv("API_KEY", "")
BENCHMARK = "BangaloreTrafficEnv"

MAX_STEPS = 100
SUCCESS_SCORE_THRESHOLD = 0.5   # normalised score in [0, 1]

# Worst-case per step: 4 lanes × ~20 cars = 80 waiting → reward ≈ −80
# Best case: reward = 0  →  score = 1.0
MAX_PENALTY_PER_STEP = 80.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an intelligent traffic light controller at a busy Bangalore intersection.
    You observe queue lengths for four lanes: [North, South, East, West].
    You must choose ONE action each step:
        0 → Give Green light to North-South lanes
        1 → Give Green light to East-West lanes
    Always choose the action that clears the longest waiting queues.
    Respond with ONLY a single digit: 0 or 1. No explanation.
    """
).strip()


# ── Mandatory log helpers (mirror the sample exactly) ────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM decision ─────────────────────────────────────────────────────────────
def build_user_prompt(step: int, state: list, last_reward: float, task_id: str) -> str:
    ns_queue = state[0] + state[1]
    ew_queue = state[2] + state[3]
    return textwrap.dedent(
        f"""
        Task: {task_id}
        Step: {step}
        Traffic queues → North: {state[0]}, South: {state[1]}, East: {state[2]}, West: {state[3]}
        NS total: {ns_queue}  |  EW total: {ew_queue}
        Last reward: {last_reward:.2f}
        Choose action 0 (NS green) or 1 (EW green). Reply with ONLY 0 or 1.
        """
    ).strip()


def get_llm_action(
    client: OpenAI,
    step: int,
    state: list,
    last_reward: float,
    task_id: str,
) -> int:
    """Call the LLM via OpenAI client and parse a 0/1 action."""
    user_prompt = build_user_prompt(step, state, last_reward, task_id)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=5,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return int(text[0]) if text and text[0] in ("0", "1") else _fallback_action(state)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return _fallback_action(state)


def _fallback_action(state: list) -> int:
    """Greedy heuristic: green the side with more waiting cars."""
    return 0 if (state[0] + state[1]) >= (state[2] + state[3]) else 1


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def env_reset(task_id: str) -> list:
    res = requests.post(
        f"{ENV_URL}/reset",
        json={"task": task_id},
        timeout=15,
    )
    res.raise_for_status()
    return res.json().get("state", [0, 0, 0, 0, 0, 0])


def env_step(action: int):
    res = requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        timeout=15,
    )
    res.raise_for_status()
    data = res.json()
    return (
        data.get("state",  [0, 0, 0, 0, 0, 0]),
        float(data.get("reward", 0.0)),
        bool(data.get("done",   False)),
        data.get("score",  0.0),   # server-side normalised score
        data.get("info",   {}),
    )


# ── Main episode loop ─────────────────────────────────────────────────────────
def run_inference(task_id: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score        = 0.001
    success      = False
    state        = [0, 0, 0, 0, 0, 0]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Reset ──────────────────────────────────────────────
        state = env_reset(task_id)
        last_reward = 0.0

        # ── Step loop ──────────────────────────────────────────
        for step in range(1, MAX_STEPS + 1):
            action = get_llm_action(client, step, state, last_reward, task_id)
            error_msg: Optional[str] = None

            try:
                state, reward, done, srv_score, info = env_step(action)
            except Exception as exc:
                error_msg = str(exc)
                reward     = 0.0
                done       = True
                srv_score  = 0.0
                success    = False

            rewards.append(reward)
            steps_taken  = step
            last_reward  = reward

            log_step(step=step, action=str(action), reward=reward, done=done, error=error_msg)

            if done or error_msg:
                break

        # ── Compute final normalised score ────────────────────
        # Each step: best reward = 0, worst = -MAX_PENALTY_PER_STEP
        # Shift so 0 → best (1.0), worst → floor (0.0)
        total_raw   = sum(rewards)
        worst_total = -MAX_PENALTY_PER_STEP * steps_taken
        if worst_total != 0:
            score = (total_raw - worst_total) / (0 - worst_total)
        else:
            score = 1.0
        score   = float(min(max(score, 0.001), 0.999))   # :.3f prints "0.001"–"0.999", never "0.000" or "1.000"
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode failed: {exc}", flush=True)
        success = False
        score   = 0.001   # must not be 0.0 — platform requires strictly > 0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────
ALL_TASKS = ["rush_hour_control", "off_peak_control", "sustained_flow"]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Grader passed a specific task as argument
        run_inference(task_id=sys.argv[1])
    else:
        # Always run all 3 — never let an env var short-circuit this
        for task_id in ALL_TASKS:
            run_inference(task_id=task_id)