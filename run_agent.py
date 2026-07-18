"""
Thin wrapper around inference.py — kept for backwards compatibility.
All logic lives in inference.run_inference().
"""
import sys

from inference import run_inference, ALL_TASKS

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_inference(task_id=sys.argv[1])
    else:
        for task_id in ALL_TASKS:
            run_inference(task_id=task_id)
