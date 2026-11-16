"""run_episode with a fake environment: checks that every policy type goes through the same path."""

import numpy as np

from tlcs.baselines import FixedTimeController
from tlcs.env import EnvStats
from tlcs.episode import run_episode


class FakeEnv:
    """Minimal Environment double recording how it was driven."""

    def __init__(self, max_steps: int = 50, green_duration: int = 10) -> None:
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.step = 0
        self.calls: list[tuple[str, int, int | None]] = []
        self.waits = iter([5.0, 3.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def generate_routefile(self, seed: int) -> None:
        self.seed = seed

    def activate(self) -> None: ...

    def deactivate(self) -> None: ...

    def is_over(self) -> bool:
        return self.step >= self.max_steps

    def get_state(self) -> np.ndarray:
        return np.zeros(80)

    def get_current_action(self) -> int:
        return 2

    def _run(self, n: int) -> list[EnvStats]:
        n = min(n, self.max_steps - self.step)
        self.step += n
        return [EnvStats(queue_length=1)] * n

    def execute(self, action: int, green_duration: int | None = None) -> list[EnvStats]:
        self.calls.append(("execute", action, green_duration))
        return self._run(self.green_duration if green_duration is None else green_duration)

    def advance(self, duration: int) -> list[EnvStats]:
        self.calls.append(("advance", -1, duration))
        return self._run(duration)

    def get_cumulated_waiting_time(self) -> float:
        return next(self.waits)


class ConstantPolicy:
    def choose_action(self, state: np.ndarray) -> int:  # noqa: ARG002
        return 1


def test_plain_policy_uses_default_green() -> None:
    env = FakeEnv()
    history, stats = run_episode(env, ConstantPolicy(), seed=3)  # type: ignore[arg-type]
    assert env.seed == 3
    assert env.calls == [("execute", 1, None)] * 5
    assert [r.step for r in history] == [10, 20, 30, 40, 50]
    # reward = previous wait - current wait
    assert [r.reward for r in history] == [-5.0, 2.0, -5.0, 8.0, 0.0]
    assert len(stats) == env.max_steps


def test_fixed_time_policy_passes_its_green_duration() -> None:
    env = FakeEnv(max_steps=60)
    history, _ = run_episode(env, FixedTimeController([20, 5, 20, 5]), seed=0)  # type: ignore[arg-type]
    assert env.calls == [
        ("execute", 0, 20),
        ("execute", 1, 5),
        ("execute", 2, 20),
        ("execute", 3, 5),
        ("execute", 0, 20),
    ]
    assert [r.step for r in history] == [20, 25, 45, 50, 60]


def test_no_policy_lets_sumo_control_the_light() -> None:
    env = FakeEnv(max_steps=25)
    history, stats = run_episode(env, None, seed=0)  # type: ignore[arg-type]
    assert env.calls == [("advance", -1, 10)] * 3
    assert [r.action for r in history] == [2, 2, 2]  # observed phase, not chosen
    assert [r.step for r in history] == [10, 20, 25]
    assert len(stats) == 25
