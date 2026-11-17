"""Test every checkpoint of a training run on one scenario and plot test results vs episodes.

Usage:
    uv run python scripts/eval_checkpoints.py --run model/koteshwor-dqn-1000 \
        --settings-file settings/koteshwor_testing_medium.yaml --test-name test-medium \
        --reference "DQN 100 ep=model/koteshwor/test" --reference "fixed=baselines/koteshwor/fixed"

Each checkpoint (`<run>/checkpoints/episode_NNNN/`) is tested with `tlcs test` logic into
`<checkpoint>/<test-name>/` (reused if it already exists, unless --rerun). Writes
`<run>/checkpoints/<test-name>_eval.csv` and `<run>/checkpoints/<test-name>_eval.png`, and prints
the best checkpoint by average waiting time per vehicle.
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

from tlcs.compare import SERIES_COLORS, load_run, metric_values
from tlcs.constants import CHECKPOINTS_FOLDER
from tlcs.main import testing_session

MAX_TICKS = 6  # label every other checkpoint when there are more
CHECKPOINT_PATTERN = re.compile(r"episode_(\d+)$")
METRICS = (
    ("wait", "Avg waiting time per vehicle (s)"),
    ("queue", "Mean queue length (vehicles)"),
    ("throughput", "Throughput (vehicles finished)"),
)


def find_checkpoints(run: Path) -> list[tuple[int, Path]]:
    """Checkpoint folders of a run as (episode, path), sorted by episode."""
    found = []
    for path in (run / CHECKPOINTS_FOLDER).iterdir():
        match = CHECKPOINT_PATTERN.match(path.name)
        if path.is_dir() and match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def main() -> None:  # noqa: C901, PLR0915
    """Parse arguments, test the checkpoints and save the table and plot."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="Training run folder")
    parser.add_argument("--settings-file", type=Path, required=True, help="Testing settings")
    parser.add_argument("--test-name", default="test", help="Test folder inside each checkpoint")
    parser.add_argument("--rerun", action="store_true", help="Re-test checkpoints already tested")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR",
        help="Test run drawn as a horizontal reference line (repeatable)",
    )
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.run)
    if not checkpoints:
        parser.error(f"No checkpoints in {args.run / CHECKPOINTS_FOLDER}")

    rows = []
    for episode, path in checkpoints:
        test_path = path / args.test_name
        if args.rerun or not (test_path / "tripinfo.xml").exists():
            print(f"Testing checkpoint after {episode} episodes ...")
            testing_session(
                settings_file=args.settings_file, model_path=path, test_name=args.test_name
            )
        run = load_run(test_path, f"episode {episode}")
        wait, queue, throughput = (metric_values([run], m)[0] for m, _ in METRICS)
        rows.append({"episode": episode, "wait": wait, "queue": queue, "throughput": throughput})

    out_dir = args.run / CHECKPOINTS_FOLDER
    csv_file = out_dir / f"{args.test_name}_eval.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "wait", "queue", "throughput"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()})

    references = []
    for ref in args.reference:
        label, _, ref_path = ref.partition("=")
        ref_run = load_run(Path(ref_path), label)
        references.append((label, [metric_values([ref_run], m)[0] for m, _ in METRICS]))

    episodes = [row["episode"] for row in rows]
    plt.rcParams.update({"font.size": 11})  # tests set a global size of 24 via tlcs.plots
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 4.8))
    for i, (ax, (metric, ylabel)) in enumerate(zip(axes, METRICS, strict=True)):
        ax.plot(
            episodes,
            [row[metric] for row in rows],
            "o-",
            color=SERIES_COLORS[0],
            lw=2,
            label="DQN checkpoint",
        )
        for (label, values), color in zip(references, SERIES_COLORS[1:], strict=False):
            if values[i] is not None:
                ax.axhline(values[i], color=color, ls="--", lw=1.5, label=label)
        ax.set_xlabel("Training episodes")
        ax.set_ylabel(ylabel)
        ax.set_xticks(episodes[1::2] if len(episodes) > MAX_TICKS else episodes)
        ax.grid(visible=True, color="#e4e3df")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Checkpoints tested on {args.settings_file.name} (fixed seed)")
    fig.tight_layout()
    png_file = out_dir / f"{args.test_name}_eval.png"
    fig.savefig(png_file, dpi=120)

    best = min(rows, key=lambda r: r["wait"] if r["wait"] is not None else float("inf"))
    final = rows[-1]["episode"]
    which = "the final one" if best["episode"] == final else f"earlier than the final ({final})"
    print(f"Saved {csv_file} and {png_file}")
    for row in rows:
        print(
            f"  episode {row['episode']:>5}: wait {row['wait']:.2f} s, queue {row['queue']:.2f}, "
            f"throughput {row['throughput']:.0f}"
        )
    print(f"Best checkpoint by average waiting time: episode {best['episode']} ({which})")
    best_path = out_dir / f"episode_{best['episode']:04d}"
    print(f"BEST={best_path}")


if __name__ == "__main__":
    main()
