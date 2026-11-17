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
    "demand",
    "avg_wait_per_vehicle_s",
    "avg_entry_delay_s",
    "mean_queue_length",
    "max_queue_length",
    "throughput",
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
    # Vehicles that finished their trip during the episode (None without tripinfo.xml).
    arrived: int | None = None
    # Seconds each vehicle waited to enter the network (SUMO departDelay); None without tripinfo.
    entry_delays: list[float] | None = None
    # Cars generated for the episode (`n_cars_generated` in run_info.json); None if unknown.
    demand: int | None = None


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


def parse_entry_delays(path: Path) -> list[float]:
    """Return how long each vehicle waited to enter the network (SUMO `departDelay`, s).

    A car waits to enter when its entry lane is full, i.e. the queue reaches the edge of the map.
    This time is not part of `waitingTime`.

    Args:
        path: Path to a tripinfo XML file.

    Returns:
        One delay per <tripinfo> element (0 where the attribute is missing).
    """
    root = ET.parse(path).getroot()  # noqa: S314 - file produced locally by SUMO
    return [float(trip.attrib.get("departDelay", 0)) for trip in root.iter("tripinfo")]


def count_arrived(path: Path) -> int:
    """Return how many vehicles in a SUMO tripinfo file finished their trip.

    Unfinished trips (written with `--tripinfo-output.write-unfinished`) have arrival -1.

    Args:
        path: Path to a tripinfo XML file.

    Returns:
        Number of <tripinfo> elements with a non-negative arrival time.
    """
    root = ET.parse(path).getroot()  # noqa: S314 - file produced locally by SUMO
    return sum(1 for trip in root.iter("tripinfo") if float(trip.attrib.get("arrival", -1)) >= 0)


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
    arrived = None
    entry_delays = None
    if (path / TRIPINFO_FILE).exists():
        vehicle_waits = parse_tripinfo(path / TRIPINFO_FILE)
        arrived = count_arrived(path / TRIPINFO_FILE)
        entry_delays = parse_entry_delays(path / TRIPINFO_FILE)
    else:
        logger.warning(f"{label}: no {TRIPINFO_FILE}; average waiting time unavailable")

    controller = "unknown"
    demand = None
    if (path / RUN_INFO_FILE).exists():
        info = json.loads((path / RUN_INFO_FILE).read_text(encoding="utf-8"))
        controller = str(info.get("controller", "unknown"))
        if info.get("n_cars_generated") is not None:
            demand = int(info["n_cars_generated"])

    return RunData(
        label=label,
        path=path,
        controller=controller,
        rewards=rewards,
        reward_steps=reward_steps,
        queue=queue,
        vehicle_waits=vehicle_waits,
        arrived=arrived,
        entry_delays=entry_delays,
        demand=demand,
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
        "demand": run.demand if run.demand is not None else "n/a",
        "avg_wait_per_vehicle_s": float(np.mean(waits)) if waits else math.nan,
        "mean_queue_length": float(np.mean(run.queue)) if run.queue else math.nan,
        "max_queue_length": float(max(run.queue)) if run.queue else math.nan,
        "avg_entry_delay_s": float(np.mean(run.entry_delays)) if run.entry_delays else math.nan,
        "throughput": run.arrived if run.arrived is not None else "n/a",
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


def metric_values(runs: list[RunData], metric: str) -> list[float | None]:
    """Value of a bar metric for each run (None where the data is missing).

    Args:
        runs: Loaded runs.
        metric: "wait" (average waiting time per vehicle), "queue" (mean queue length) or
            "throughput" (vehicles that finished their trip).

    Returns:
        One value per run, in order.
    """
    values: list[float | None] = []
    for run in runs:
        if metric == "wait":
            values.append(float(np.mean(run.vehicle_waits)) if run.vehicle_waits else None)
        elif metric == "queue":
            values.append(float(np.mean(run.queue)) if run.queue else None)
        elif metric == "throughput":
            values.append(float(run.arrived) if run.arrived is not None else None)
        else:
            msg = f"Unknown metric '{metric}'"
            raise ValueError(msg)
    return values


# metric -> (file name, axis label, title, value format)
BAR_METRICS = {
    "wait": (
        "compare_wait.png",
        "Average waiting time per vehicle (s)",
        "Average waiting time per vehicle",
        "{:.1f}",
    ),
    "queue": (
        "compare_queue_mean.png",
        "Mean queue length (vehicles)",
        "Mean queue length",
        "{:.1f}",
    ),
    "throughput": (
        "compare_throughput.png",
        "Vehicles that finished their trip",
        "Throughput",
        "{:.0f}",
    ),
}


def _annotate(ax: plt.Axes, bar: object, value: float | None, fmt: str, fontsize: int) -> None:
    """Write the value above a bar ("n/a" when missing)."""
    ax.annotate(
        fmt.format(value) if value is not None else "n/a",
        (bar.get_x() + bar.get_width() / 2, value or 0.0),  # type: ignore[attr-defined]
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        fontsize=fontsize,
        color="#2b2a27",
    )


def plot_bars(runs: list[RunData], metric: str, out_file: Path) -> None:
    """Bar chart of one metric, one bar per run."""
    _, ylabel, title, fmt = BAR_METRICS[metric]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    values = metric_values(runs, metric)
    bars = ax.bar(
        [r.label for r in runs],
        [v or 0.0 for v in values],
        color=SERIES_COLORS[: len(runs)],
        width=0.6,
    )
    for bar, value in zip(bars, values, strict=True):
        _annotate(ax, bar, value, fmt, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def plot_wait(runs: list[RunData], out_file: Path) -> None:
    """Bar chart of the average waiting time per vehicle, one bar per run."""
    plot_bars(runs, "wait", out_file)


def series_and_groups(runs: list[RunData]) -> tuple[list[str], list[int | None]]:
    """Series (run labels, first-seen order) and demand groups (ascending, unknown last)."""
    series = list(dict.fromkeys(r.label for r in runs))
    groups: list[int | None] = sorted({r.demand for r in runs if r.demand is not None})
    if any(r.demand is None for r in runs):
        groups.append(None)
    return series, groups


def plot_grouped_bars(runs: list[RunData], metric: str, out_file: Path) -> None:
    """Grouped bar chart: one group per demand level, one coloured bar per label."""
    _, ylabel, title, fmt = BAR_METRICS[metric]
    series, groups = series_and_groups(runs)
    values = {
        (run.label, run.demand): value
        for run, value in zip(runs, metric_values(runs, metric), strict=True)
    }

    fig, ax = plt.subplots(figsize=(max(10.0, 1.1 * len(groups) * len(series)), 5.5))
    width = 0.8 / len(series)
    x = np.arange(len(groups))
    for i, (name, color) in enumerate(zip(series, SERIES_COLORS, strict=False)):
        present = [(j, g) for j, g in enumerate(groups) if (name, g) in values]
        vals = [values[(name, g)] for _, g in present]
        bars = ax.bar(
            [x[j] - 0.4 + width * (i + 0.5) for j, _ in present],
            [v or 0.0 for v in vals],
            width=width * 0.92,
            color=color,
            label=name,
        )
        for bar, value in zip(bars, vals, strict=True):
            _annotate(ax, bar, value, fmt, fontsize=9)
    ax.set_xticks(x, [f"{g} cars" if g is not None else "unknown" for g in groups])
    ax.set_xlabel("Demand (cars generated per episode)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} by demand")
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
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


def compare_session(
    run_paths: list[Path],
    out_path: Path,
    smooth: int = 60,
    labels: list[str] | None = None,
) -> None:
    """Load several test runs, save comparison plots and a summary CSV, and print the table.

    Runs at one demand level (at most 8): bar charts of waiting time, mean queue and
    throughput, plus the queue and cumulative-reward time series.
    Runs at several demand levels (`n_cars_generated` in run_info.json): one grouped bar chart
    per metric instead, one group per demand and one bar per label (at most 8 labels). Give the
    runs of one controller the same label at every demand.

    Args:
        run_paths: Folders written by `tlcs test` or `tlcs baseline`.
        out_path: Folder where plots and summary.csv are written.
        smooth: Moving-average window (steps) for the queue plot; 1 disables smoothing.
        labels: Display label per run; None derives labels from the folder names.
    """
    if labels is None:
        labels = make_labels(run_paths)
    elif len(labels) != len(run_paths):
        msg = f"Got {len(labels)} labels for {len(run_paths)} runs"
        raise ValueError(msg)

    runs = [load_run(path, label) for path, label in zip(run_paths, labels, strict=True)]
    grouped = len({r.demand for r in runs}) > 1

    if grouped:
        keys = [(r.label, r.demand) for r in runs]
        if len(set(keys)) != len(keys):
            msg = "Two runs have the same label and demand; give them different labels"
            raise ValueError(msg)
        n_series = len(series_and_groups(runs)[0])
        if n_series > MAX_RUNS:
            msg = f"At most {MAX_RUNS} labels can be compared at once (got {n_series})"
            raise ValueError(msg)
    elif len(runs) > MAX_RUNS:
        msg = f"At most {MAX_RUNS} runs can be compared at once (got {len(runs)})"
        raise ValueError(msg)

    out_path.mkdir(parents=True, exist_ok=True)
    rows = [summarize(run) for run in runs]
    write_summary_csv(rows, out_path / SUMMARY_FILE)

    with plt.rc_context({"font.size": 12}):
        for metric, (file_name, *_) in BAR_METRICS.items():
            if grouped:
                plot_grouped_bars(runs, metric, out_path / file_name)
            else:
                plot_bars(runs, metric, out_path / file_name)
        if not grouped:
            plot_queue(runs, out_path / "compare_queue.png", smooth=smooth)
            plot_cumulative_reward(runs, out_path / "compare_reward.png")

    print_summary(rows)
    logger.info(f"Comparison saved at: {out_path}")
