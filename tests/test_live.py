"""Live side-by-side runner: layout, controller parsing, and equivalence with run_episode."""

import numpy as np
import pytest

from tlcs.baselines import FixedTimeController, RandomController
from tlcs.env import EnvStats, build_phase_plan
from tlcs.episode import Policy, run_episode
from tlcs.live import ALL_CONTROLLERS, StepRunner, parse_controllers, tile_windows

# --- phase plan (shared yellow-insertion rule) ------------------------------------------------


GREENS = ("GGrr", "rrGG", "GrGr")
YELLOWS = ("yyrr", "rryy", "yryr")


def _plan(current: int, action: int) -> list[tuple[str, int]]:
    return build_phase_plan(current, action, GREENS, YELLOWS, yellow_duration=4, green_duration=10)


def test_phase_plan_inserts_yellow_only_on_change() -> None:
    assert _plan(0, 1) == [("yyrr", 4), ("rrGG", 10)]  # change: yellow of the current green first
    assert _plan(1, 1) == [("rrGG", 10)]  # same green: just extend it
    assert _plan(-1, 2) == [("GrGr", 10)]  # no green active (start / yellow): no extra yellow


def test_phase_plan_yellow_keeps_links_that_stay_green() -> None:
    # Link 0 is green in both "GGrr" and "GrGr": it stays green, the others clear.
    assert _plan(0, 2) == [("yyrr", 4), ("GrGr", 10)]
    yellows_keeping_link_0 = ("Gyrr", "rryy", "yryr")
    plan = build_phase_plan(0, 2, GREENS, yellows_keeping_link_0, 4, 10)
    assert plan[0] == ("Gyrr", 4)
    plan = build_phase_plan(0, 1, GREENS, yellows_keeping_link_0, 4, 10)
    assert plan[0] == ("yyrr", 4)  # link 0 is red in "rrGG", so it must turn yellow


# --- layout and parsing -----------------------------------------------------------------------


def test_tile_windows_grid_keeps_title_bars_on_screen() -> None:
    tiles = tile_windows(6, 1800, 1000, title_bar=40)
    assert len(tiles) == 6
    assert tiles[0] == (0, 40, 600, 460)
    assert tiles[5] == (1200, 540, 600, 460)
    assert all(y >= 40 and y + h <= 1000 for _, y, _, h in tiles)


def test_tile_windows_single_window_fills_screen() -> None:
    assert tile_windows(1, 1000, 800, title_bar=0) == [(0, 0, 1000, 800)]


def test_parse_controllers() -> None:
    assert parse_controllers("DQN, lqf,lqf ,actuated") == ["dqn", "lqf", "actuated"]
    assert tuple(parse_controllers(",".join(ALL_CONTROLLERS))) == ALL_CONTROLLERS
    with pytest.raises(ValueError, match="Unknown"):
        parse_controllers("dqn,greedy")
    with pytest.raises(ValueError, match="empty"):
        parse_controllers(" , ")


# --- StepRunner reproduces run_episode --------------------------------------------------------


class PhaseFakeEnv:
    """Environment double with a real phase state and step-dependent waiting/queues.

    `execute` follows the same plan semantics as Environment.execute.
    """

    def __init__(self, max_steps: int = 137) -> None:
        self.max_steps = max_steps
        self.green_duration = 10
        self.yellow_duration = 4
        self.step = 0
        self.signal = GREENS[0]
        self.phase_log: list[str] = []

    def generate_routefile(self, seed: int) -> None: ...

    def activate(self) -> None: ...

    def deactivate(self) -> None: ...

    def is_over(self) -> bool:
        return self.step >= self.max_steps

    def get_state(self) -> np.ndarray:
        return np.full(80, float(self.step % 7))

    def get_current_action(self) -> int:
        return GREENS.index(self.signal) if self.signal in GREENS else -1

    def get_cumulated_waiting_time(self) -> float:
        return float((self.step * 37) % 101)

    def plan_action(self, action: int, green_duration: int | None = None) -> list[tuple[str, int]]:
        green = self.green_duration if green_duration is None else green_duration
        return build_phase_plan(
            self.get_current_action(), action, GREENS, YELLOWS, self.yellow_duration, green
        )

    def set_signal(self, signal: str) -> None:
        self.signal = signal

    def advance(self, duration: int) -> list[EnvStats]:
        stats = []
        for _ in range(min(duration, self.max_steps - self.step)):
            self.step += 1
            self.phase_log.append(self.signal)
            stats.append(EnvStats(queue_length=(self.step * 13 + len(set(self.signal))) % 9))
        return stats

    def execute(self, action: int, green_duration: int | None = None) -> list[EnvStats]:
        stats: list[EnvStats] = []
        for signal, duration in self.plan_action(action, green_duration):
            if self.is_over():
                break
            self.set_signal(signal)
            stats.extend(self.advance(duration))
        return stats


def _make_policy(kind: str) -> Policy | None:
    return {
        "fixed": FixedTimeController([20, 5, 15]),
        "random": RandomController(seed=3, num_actions=3),
        "actuated": None,
    }[kind]


@pytest.mark.parametrize("kind", ["fixed", "random", "actuated"])
def test_step_runner_matches_run_episode(kind: str) -> None:
    reference_env = PhaseFakeEnv()
    ref_history, ref_stats = run_episode(reference_env, _make_policy(kind), seed=0)  # type: ignore[arg-type]

    live_env = PhaseFakeEnv()
    runner = StepRunner(kind, live_env, _make_policy(kind))  # type: ignore[arg-type]
    for _ in range(live_env.max_steps + 5):  # extra steps must be no-ops
        runner.step()
    runner.finish()

    assert live_env.phase_log == reference_env.phase_log
    assert runner.env_stats == ref_stats
    assert [(r.action, r.reward, r.step) for r in runner.history] == [
        (r.action, r.reward, r.step) for r in ref_history
    ]
    assert all(
        np.array_equal(a.state, b.state) for a, b in zip(runner.history, ref_history, strict=True)
    )
