"""Compare saved test runs (DQN and baselines) with plots and a summary table."""

import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

from tlcs.constants import REWARD_STEPS_FILE, RUN_INFO_FILE, TRIPINFO_FILE
from tlcs.logger import get_logger

logger = get_logger(__name__)

REWARD_DATA_FILE = "plot_reward_data.txt"
QUEUE_DATA_FILE = "plot_queue_data.txt"
SUMMARY_FILE = "summary.csv"

# Categorical palette, assigned in fixed order (validated for colour-vision deficiency).
SERIES_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
MAX_RUNS = len(SERIES_COLORS)

SUMMARY_COLUMNS = (
    "run",
    "controller",
    "avg_wait_per_vehicle_s",
    "mean_queue_length",
    "max_queue_length",
    "cumulative_negative_reward",
    "vehicles",
)


@dataclass
class RunData:
    """Saved results of one test run."""

    label: str
    path: Path
    controller: str
    rewards: list[float]
    reward_steps: list[int] | None
    queue: list[float]
    vehicle_waits: list[float] | None


def _read_numbers(path: Path) -> list[float]:
    """Read one number per line from a text file, skipping blank lines."""
    return [float(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_tripinfo(path: Path) -> list[float]:
    """Return the total waiting time (s) of every vehicle in a SUMO tripinfo file.

    Args:
        path: Path to a tripinfo XML file.

    Returns:
        One waiting time per <tripinfo> element (finished and unfinished trips).
    """
    root = ET.parse(path).getroot()  # noqa: S314 - file produced locally by SUMO
    return [float(trip.attrib["waitingTime"]) for trip in root.iter("tripinfo")]


def make_labels(paths: list[Path]) -> list[str]:
    """Build short, unique labels for run folders.

    Uses the folder name, falling back to "parent/name" when folder names collide.

    Args:
        paths: Run folders.

    Returns:
        One label per path, in order.
    """
    names = [p.name for p in paths]
    return [f"{p.parent.name}/{p.name}" if names.count(p.name) > 1 else p.name for p in paths]


def load_run(path: Path, label: str) -> RunData:
    """Load the saved data of a test run folder.

    Args:
        path: Folder written by `tlcs test` or `tlcs baseline`.
        label: Display label for the run.

    Returns:
        The loaded run data. Optional files that are missing (older runs) become None.
    """
    for required in (REWARD_DATA_FILE, QUEUE_DATA_FILE):
        if not (path / required).exists():
            msg = f"'{path}' does not look like a test run folder (missing {required})"
            raise FileNotFoundError(msg)

    rewards = _read_numbers(path / REWARD_DATA_FILE)
    queue = _read_numbers(path / QUEUE_DATA_FILE)

    reward_steps = None
    if (path / REWARD_STEPS_FILE).exists():
        reward_steps = [int(v) for v in _read_numbers(path / REWARD_STEPS_FILE)]
    else:
        logger.warning(f"{label}: no {REWARD_STEPS_FILE}; rewards plotted by action index")

    vehicle_waits = None
    if (path / TRIPINFO_FILE).exists():
        vehicle_waits = parse_tripinfo(path / TRIPINFO_FILE)
    else:
        logger.warning(f"{label}: no {TRIPINFO_FILE}; average waiting time unavailable")

    controller = "unknown"
    if (path / RUN_INFO_FILE).exists():
        info = json.loads((path / RUN_INFO_FILE).read_text(encoding="utf-8"))
        controller = str(info.get("controller", "unknown"))

    return RunData(
        label=label,
        path=path,
        controller=controller,
        rewards=rewards,
        reward_steps=reward_steps,
        queue=queue,
        vehicle_waits=vehicle_waits,
    )


def summarize(run: RunData) -> dict[str, str | float | int]:
    """Compute the summary row for one run.

    Args:
        run: Loaded run data.

    Returns:
        Mapping column name -> value (NaN where data is missing).
    """
    waits = run.vehicle_waits
    return {
        "run": run.label,
        "controller": run.controller,
        "avg_wait_per_vehicle_s": float(np.mean(waits)) if waits else math.nan,
        "mean_queue_length": float(np.mean(run.queue)) if run.queue else math.nan,
        "max_queue_length": float(max(run.queue)) if run.queue else math.nan,
        "cumulative_negative_reward": float(sum(r for r in run.rewards if r < 0)),
        "vehicles": len(waits) if waits is not None else 0,
    }


def write_summary_csv(rows: list[dict[str, str | float | int]], path: Path) -> None:
    """Write summary rows to a CSV file (floats rounded to 2 decimals)."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()})


def print_summary(rows: list[dict[str, str | float | int]]) -> None:
    """Print the summary rows as a table."""
    table = Table(title="Run comparison (means over the test episode)")
    for col in SUMMARY_COLUMNS:
        table.add_column(col, justify="left" if col in {"run", "controller"} else "right")
    for row in rows:
        table.add_row(
            *(
                f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c])
                for c in SUMMARY_COLUMNS
            )
        )
    Console().print(table)


def rolling_mean(values: list[float], window: int) -> NDArray:
    """Trailing moving average (window shrinks at the start); window <= 1 returns the input."""
    data = np.asarray(values, dtype=float)
    if window <= 1 or data.size == 0:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    idx = np.arange(1, data.size + 1)
    start = np.maximum(idx - window, 0)
    return (cumsum[idx] - cumsum[start]) / (idx - start)


def _style_axes(ax: plt.Axes) -> None:
    """Recessive grid and axes."""
    ax.grid(visible=True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#55544f")


def plot_wait(runs: list[RunData], out_file: Path) -> None:
    """Bar chart of the average waiting time per vehicle, one bar per run."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [r.label for r in runs]
    values = [float(np.mean(r.vehicle_waits)) if r.vehicle_waits else 0.0 for r in runs]
    bars = ax.bar(labels, values, color=SERIES_COLORS[: len(runs)], width=0.6)
    for bar, run, value in zip(bars, runs, values, strict=True):
        text = f"{value:.1f}" if run.vehicle_waits else "n/a"
        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color="#2b2a27",
        )
    ax.set_ylabel("Average waiting time per vehicle (s)")
    ax.set_title("Average waiting time per vehicle")
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def plot_queue(runs: list[RunData], out_file: Path, smooth: int) -> None:
    """Line chart of queue length over simulation time, one line per run."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for run, color in zip(runs, SERIES_COLORS, strict=False):
        ax.plot(rolling_mean(run.queue, smooth), color=color, linewidth=2, label=run.label)
    ax.set_xlabel("Simulation step (s)")
    ax.set_ylabel("Queue length (vehicles)")
    suffix = f" ({smooth}-step moving average)" if smooth > 1 else ""
    ax.set_title(f"Queue length{suffix}")
    ax.margins(x=0)
    _style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def cumulative_negative_reward(rewards: list[float]) -> NDArray:
    """Running sum of the negative rewards only (the quantity of the training reward plot).

    The plain running sum of the reward is not used: the reward is the change in waiting time,
    so its sum telescopes to minus the waiting time still on the approaches, which is ~0 once
    the network empties.
    """
    return np.cumsum(np.minimum(np.asarray(rewards, dtype=float), 0.0))


def plot_cumulative_reward(runs: list[RunData], out_file: Path) -> None:
    """Line chart of cumulative negative reward over simulation time, one line per run."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    by_index = any(r.reward_steps is None for r in runs)
    for run, color in zip(runs, SERIES_COLORS, strict=False):
        x = (
            np.arange(1, len(run.rewards) + 1)
            if run.reward_steps is None or by_index
            else np.asarray(run.reward_steps)
        )
        ax.plot(
            x, cumulative_negative_reward(run.rewards), color=color, linewidth=2, label=run.label
        )
    ax.set_xlabel("Action step" if by_index else "Simulation step (s)")
    ax.set_ylabel("Cumulative negative reward")
    ax.set_title("Cumulative negative reward")
    ax.margins(x=0)
    _style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def compare_session(run_paths: list[Path], out_path: Path, smooth: int = 60) -> None:
    """Load several test runs, save comparison plots and a summary CSV, and print the table.

    Args:
        run_paths: Folders written by `tlcs test` or `tlcs baseline`.
        out_path: Folder where plots and summary.csv are written.
        smooth: Moving-average window (steps) for the queue plot; 1 disables smoothing.
    """
    if len(run_paths) > MAX_RUNS:
        msg = f"At most {MAX_RUNS} runs can be compared at once (got {len(run_paths)})"
        raise ValueError(msg)

    labels = make_labels(run_paths)
    runs = [load_run(path, label) for path, label in zip(run_paths, labels, strict=True)]

    out_path.mkdir(parents=True, exist_ok=True)
    rows = [summarize(run) for run in runs]
    write_summary_csv(rows, out_path / SUMMARY_FILE)

    with plt.rc_context({"font.size": 12}):
        plot_wait(runs, out_path / "compare_wait.png")
        plot_queue(runs, out_path / "compare_queue.png", smooth=smooth)
        plot_cumulative_reward(runs, out_path / "compare_reward.png")

    print_summary(rows)
    logger.info(f"Comparison saved at: {out_path}")
