"""Plot the learning curves of several training runs side by side.

Usage:
    uv run python scripts/plot_learning_curves.py --runs model/koteshwor model/koteshwor-dqn-1000 \
        --labels "100 episodes" "1000 episodes" --out docs/figures/learning_curves.png

One column per run, one row per metric (cumulative negative reward and average queue length per
training episode, from the run's plot_*_data.txt). Rows share the y axis so runs are comparable.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tlcs.compare import rolling_mean

METRICS = (
    ("plot_reward_data.txt", "Cumulative negative reward"),
    ("plot_queue_data.txt", "Average queue length (vehicles)"),
)


def main() -> None:
    """Parse arguments and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smooth", type=int, default=10, help="Moving-average window (episodes)")
    parser.add_argument(
        "--clip",
        action="store_true",
        help=(
            "Limit each row's y axis to the 1st-99th percentile of all runs, so a few extreme "
            "episodes do not flatten the curves; off-scale episodes are marked and counted"
        ),
    )
    args = parser.parse_args()
    labels = args.labels or [run.name for run in args.runs]
    data_by_run = [[np.loadtxt(run / file_name) for file_name, _ in METRICS] for run in args.runs]
    limits = []
    for row in range(len(METRICS)):
        values = np.concatenate([run_data[row] for run_data in data_by_run])
        low, high = np.percentile(values, [1, 99])
        pad = 0.05 * (high - low)
        limits.append((low - pad, high + pad))

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(
        len(METRICS),
        len(args.runs),
        figsize=(6 * len(args.runs), 4 * len(METRICS)),
        sharey="row",
        squeeze=False,
    )
    for col, label in enumerate(labels):
        for row, (_, ylabel) in enumerate(METRICS):
            data = data_by_run[col][row]
            episodes = np.arange(1, data.size + 1)
            ax = axes[row][col]
            ax.plot(episodes, data, color="#9fc3ef", lw=1, label="per episode")
            ax.plot(
                episodes,
                rolling_mean(list(data), args.smooth),
                color="#2a78d6",
                lw=2,
                label=f"{args.smooth}-episode average",
            )
            if args.clip:
                low, high = limits[row]
                ax.set_ylim(low, high)
                off = (data < low) | (data > high)
                if off.any():
                    edge = np.where(data[off] > high, high, low)
                    ax.plot(
                        episodes[off],
                        edge,
                        "v" if edge[0] < high else "^",
                        color="#e34948",
                        ms=8,
                        clip_on=False,
                        label="off-scale episode",
                    )
                    ax.annotate(
                        f"{int(off.sum())} off-scale: episode "
                        + ", ".join(str(e) for e in episodes[off]),
                        (0.98, 0.04),
                        xycoords="axes fraction",
                        ha="right",
                        color="#e34948",
                        fontsize=9,
                    )
            ax.set_xlabel("Training episode")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(label)
            ax.grid(visible=True, color="#e4e3df")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
