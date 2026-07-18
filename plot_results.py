"""Render results_comparison.png from results.csv (matplotlib only).

One panel per task (own y-scale — rush_hour queues are ~15x the others),
bars = mean avg-queue-per-lane per controller, sustained_flow panel annotated
with success rates.
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt

CONTROLLERS = ["ppo", "greedy", "fixed_1", "fixed_2", "fixed_3", "fixed_5", "fixed_10"]
TASKS = ["rush_hour_control", "off_peak_control", "sustained_flow"]
TASK_LABELS = {
    "rush_hour_control": "rush_hour (hard)",
    "off_peak_control": "off_peak (easy)",
    "sustained_flow": "sustained_flow (medium)",
}
# PPO blue, greedy aqua, fixed-timer family as a recessive gray ramp
COLORS = {
    "ppo": "#2a78d6",
    "greedy": "#1baf7a",
    "fixed_1": "#b5b3ac",
    "fixed_2": "#a09e97",
    "fixed_3": "#8b8981",
    "fixed_5": "#73716a",
    "fixed_10": "#5c5a54",
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def load(path="results.csv"):
    queues = defaultdict(list)   # (task, controller) -> [avg_queue]
    success = defaultdict(list)  # (task, controller) -> [bool]
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["task"], row["controller"])
            queues[key].append(float(row["avg_queue_per_lane"]))
            success[key].append(row["success"] == "True")
    return queues, success


def main():
    queues, success = load()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="white")

    for ax, task in zip(axes, TASKS):
        means = [sum(queues[(task, c)]) / len(queues[(task, c)]) for c in CONTROLLERS]
        bars = ax.bar(
            range(len(CONTROLLERS)),
            means,
            width=0.62,
            color=[COLORS[c] for c in CONTROLLERS],
            zorder=3,
        )
        for bar, mean in zip(bars, means):
            ax.annotate(
                f"{mean:.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK,
            )
        if task == "sustained_flow":
            ax.axhline(5.0, color="#d03b3b", linewidth=1, linestyle="--", zorder=2)
            ax.annotate("threshold = 5", (0.02, 4.6), fontsize=8, color="#d03b3b")
            for bar, c in zip(bars, CONTROLLERS):
                rate = sum(success[(task, c)]) / len(success[(task, c)])
                ax.annotate(
                    f"{rate:.0%}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8, color=MUTED,
                )
            ax.margins(y=0.18)
        suffix = "  ·  % = success rate" if task == "sustained_flow" else ""
        ax.set_title(TASK_LABELS[task] + suffix, fontsize=11, color=INK)
        ax.set_xticks(range(len(CONTROLLERS)))
        ax.set_xticklabels(CONTROLLERS, rotation=35, ha="right", fontsize=8.5, color=MUTED)
        ax.tick_params(axis="y", labelsize=8.5, colors=MUTED, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")

    axes[0].set_ylabel("mean avg queue per lane (50 episodes)", fontsize=9.5, color=INK)
    fig.suptitle(
        "BangaloreTrafficEnv: PPO vs baselines — lower is better",
        fontsize=13, color=INK,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("results_comparison.png", dpi=150, bbox_inches="tight")
    print("wrote results_comparison.png")


if __name__ == "__main__":
    main()
