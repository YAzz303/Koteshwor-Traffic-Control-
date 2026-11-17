import csv
import math
from pathlib import Path

import numpy as np
import pytest

from tlcs.compare import (
    compare_session,
    count_arrived,
    cumulative_negative_reward,
    load_run,
    make_labels,
    parse_entry_delays,
    parse_tripinfo,
    rolling_mean,
    summarize,
)
from tlcs.env import EnvStats
from tlcs.episode import Record
from tlcs.main import save_test_results
from tlcs.settings import TestingSettings as Settings

TRIPINFO = """<?xml version="1.0"?>
<tripinfos>
    <tripinfo id="a" depart="0" arrival="50" waitingTime="10.00"/>
    <tripinfo id="b" depart="5" arrival="-1" waitingTime="20.00"/>
</tripinfos>
"""


def _settings(n_cars: int = 2) -> Settings:
    return Settings.model_validate(
        {
            "gui": False,
            "max_steps": 30,
            "n_cars_generated": n_cars,
            "episode_seed": 1,
            "yellow_duration": 4,
            "green_duration": 10,
            "turn_chance": 0.25,
            "gamma": 0.75,
            "sumocfg_file": "intersection/sumo_config.sumocfg",
        }
    )


def _make_run(
    tmp_path: Path, name: str, rewards: list[float], queue: list[int], n_cars: int = 2
) -> Path:
    """Write a run folder the same way `tlcs test` / `tlcs baseline` do."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("gui: false\n", encoding="utf-8")
    run = tmp_path / name
    history = [
        Record(state=np.zeros(80), action=0, reward=r, step=10 * (i + 1))
        for i, r in enumerate(rewards)
    ]
    save_test_results(
        out_path=run,
        settings_file=settings_file,
        settings=_settings(n_cars),
        controller=name,
        episode_history=history,
        env_stats=[EnvStats(queue_length=q) for q in queue],
    )
    (run / "tripinfo.xml").write_text(TRIPINFO, encoding="utf-8")
    return run


def test_parse_tripinfo_includes_unfinished(tmp_path: Path) -> None:
    path = tmp_path / "tripinfo.xml"
    path.write_text(TRIPINFO, encoding="utf-8")
    assert parse_tripinfo(path) == [10.0, 20.0]


def test_make_labels_disambiguates_duplicates() -> None:
    paths = [Path("model/run-01/test"), Path("model/run-02/test"), Path("baselines/fixed")]
    assert make_labels(paths) == ["run-01/test", "run-02/test", "fixed"]


def test_cumulative_negative_reward_ignores_positive_rewards() -> None:
    assert cumulative_negative_reward([-2, 3, -1, 0]).tolist() == [-2, -2, -3, -3]


def test_rolling_mean() -> None:
    assert rolling_mean([2, 4, 6, 8], window=2).tolist() == [2, 3, 5, 7]
    assert rolling_mean([1, 2], window=1).tolist() == [1, 2]


def test_save_then_load_roundtrip_and_summary(tmp_path: Path) -> None:
    run_path = _make_run(tmp_path, "lqf", rewards=[-4.0, 1.0, 3.0], queue=[0, 2, 4])
    run = load_run(run_path, "lqf")
    assert run.controller == "lqf"
    assert run.reward_steps == [10, 20, 30]

    row = summarize(run)
    assert row["avg_wait_per_vehicle_s"] == 15.0
    assert row["mean_queue_length"] == 2.0
    assert row["max_queue_length"] == 4.0
    assert row["cumulative_negative_reward"] == -4.0
    assert row["vehicles"] == 2
    assert row["throughput"] == 1  # one trip unfinished (arrival -1)
    assert row["demand"] == 2


def test_load_run_tolerates_old_runs_without_new_files(tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    (old / "plot_reward_data.txt").write_text("-1\n-2\n", encoding="utf-8")
    (old / "plot_queue_data.txt").write_text("1\n3\n", encoding="utf-8")
    run = load_run(old, "old")
    assert run.reward_steps is None
    assert run.vehicle_waits is None
    assert run.controller == "unknown"
    assert math.isnan(summarize(run)["avg_wait_per_vehicle_s"])  # type: ignore[arg-type]


def test_load_run_rejects_non_run_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="test run folder"):
        load_run(tmp_path, "nothing")


def test_compare_session_writes_plots_and_csv(tmp_path: Path) -> None:
    a = _make_run(tmp_path, "dqn", rewards=[-1.0, -2.0], queue=[1, 1, 2])
    b = _make_run(tmp_path, "fixed", rewards=[-3.0, 1.0], queue=[2, 3, 4])
    out = tmp_path / "cmp"

    compare_session([a, b], out, smooth=2)

    for name in ("compare_wait.png", "compare_queue.png", "compare_reward.png", "summary.csv"):
        assert (out / name).exists(), name
    with (out / "summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["run"] for r in rows] == ["dqn", "fixed"]
    assert rows[1]["mean_queue_length"] == "3.0"
    assert rows[0]["cumulative_negative_reward"] == "-3.0"


def test_count_arrived_skips_unfinished(tmp_path: Path) -> None:
    path = tmp_path / "tripinfo.xml"
    path.write_text(TRIPINFO, encoding="utf-8")
    assert count_arrived(path) == 1


def test_compare_session_groups_runs_by_demand(tmp_path: Path) -> None:
    runs, labels = [], []
    for n_cars in (2, 4):
        for name in ("dqn", "fixed"):
            runs.append(_make_run(tmp_path / str(n_cars), name, [-1.0], [1, 2], n_cars=n_cars))
            labels.append(name)
    out = tmp_path / "cmp"

    compare_session(runs, out, labels=labels)

    for name in ("compare_wait.png", "compare_queue_mean.png", "compare_throughput.png"):
        assert (out / name).exists(), name
    assert not (out / "compare_queue.png").exists()  # time series only for one demand
    with (out / "summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(r["run"], r["demand"]) for r in rows] == [
        ("dqn", "2"),
        ("fixed", "2"),
        ("dqn", "4"),
        ("fixed", "4"),
    ]


def test_compare_session_rejects_duplicate_label_and_demand(tmp_path: Path) -> None:
    a = _make_run(tmp_path / "a", "dqn", [-1.0], [1], n_cars=2)
    b = _make_run(tmp_path / "b", "dqn", [-1.0], [1], n_cars=4)
    c = _make_run(tmp_path / "c", "dqn", [-1.0], [1], n_cars=4)
    with pytest.raises(ValueError, match="same label and demand"):
        compare_session([a, b, c], tmp_path / "cmp", labels=["x", "x", "x"])


def test_compare_session_checks_label_count(tmp_path: Path) -> None:
    a = _make_run(tmp_path, "dqn", [-1.0], [1])
    with pytest.raises(ValueError, match="labels"):
        compare_session([a], tmp_path / "cmp", labels=["x", "y"])


def test_entry_delay_is_read_from_depart_delay(tmp_path: Path) -> None:
    path = tmp_path / "tripinfo.xml"
    path.write_text(
        '<tripinfos><tripinfo id="a" arrival="9" waitingTime="1" departDelay="30.00"/>'
        '<tripinfo id="b" arrival="9" waitingTime="1"/></tripinfos>',
        encoding="utf-8",
    )
    assert parse_entry_delays(path) == [30.0, 0.0]
    run = _make_run(tmp_path, "fixed", rewards=[-1.0], queue=[1])
    (run / "tripinfo.xml").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    assert summarize(load_run(run, "fixed"))["avg_entry_delay_s"] == 15.0
