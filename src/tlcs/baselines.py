"""Non-learning baseline controllers to compare the DQN agent against.

All controllers that pick phases (fixed, random, lqf, maxpressure) act through
`Environment.execute`, exactly like the DQN agent: same decision cadence, same yellow insertion
when the phase changes, same reward. The actuated baseline instead hands the light to SUMO's
built-in actuated logic and only observes.
"""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tlcs.constants import ACTUATED_PROGRAM_ID, TRIPINFO_FILE
from tlcs.env import DEFAULT_CELLS_PER_LANE, Environment
from tlcs.episode import Policy, run_episode
from tlcs.logger import get_logger
from tlcs.main import save_test_results
from tlcs.network import NetworkSpec, load_network_spec
from tlcs.settings import TestingSettings, load_testing_settings

logger = get_logger(__name__)


class ControllerName(StrEnum):
    """Baseline controllers available from the CLI."""

    FIXED = "fixed"
    RANDOM = "random"
    LQF = "lqf"
    MAXPRESSURE = "maxpressure"
    ACTUATED = "actuated"


# ---------------------------------------------------------------------------
# Pure decision logic (testable without SUMO)
# ---------------------------------------------------------------------------


def all_lanes(spec: NetworkSpec) -> list[str]:
    """Return every incoming and outgoing lane used by any action."""
    lanes: dict[str, None] = {}
    for action in range(spec.num_actions):
        in_lanes, out_lanes = spec.action_lanes(action)
        lanes.update(dict.fromkeys(in_lanes + out_lanes))
    return list(lanes)


def argmax_prefer_current(scores: Sequence[float], current_action: int) -> int:
    """Return the index of the highest score; ties keep the current action, else the lowest index.

    Keeping the current action on ties avoids pointless yellow phases.

    Args:
        scores: One score per action.
        current_action: Currently active action (-1 if unknown).

    Returns:
        The selected action index.
    """
    best = max(scores)
    if 0 <= current_action < len(scores) and scores[current_action] == best:
        return current_action
    return list(scores).index(best)


def queue_scores(halting: Mapping[str, int], spec: NetworkSpec) -> list[int]:
    """Total queued (halting) vehicles on the incoming lanes of each action.

    Args:
        halting: Mapping lane ID -> halting vehicle count.
        spec: The junction (its movements define each action's lanes).

    Returns:
        One total queue length per action.
    """
    return [
        sum(halting.get(lane, 0) for lane in spec.action_lanes(action)[0])
        for action in range(spec.num_actions)
    ]


def pressure_scores(vehicles: Mapping[str, int], spec: NetworkSpec) -> list[int]:
    """Pressure of each action: vehicles on its incoming lanes minus vehicles on its outgoing lanes.

    Args:
        vehicles: Mapping lane ID -> vehicle count.
        spec: The junction (its movements define each action's lanes).

    Returns:
        One pressure value per action.
    """
    scores = []
    for action in range(spec.num_actions):
        in_lanes, out_lanes = spec.action_lanes(action)
        upstream = sum(vehicles.get(lane, 0) for lane in in_lanes)
        downstream = sum(vehicles.get(lane, 0) for lane in out_lanes)
        scores.append(upstream - downstream)
    return scores


def build_actuated_program(spec: NetworkSpec, min_green: int, max_green: int, yellow: int) -> str:
    """Build a SUMO additional file with an actuated program for the junction.

    The program cycles through the discovered green phases in order, each followed by its
    yellow, with yellow phases of the same length the agent uses.

    Args:
        spec: The junction.
        min_green: Minimum green time (s) of each green phase.
        max_green: Maximum green time (s) of each green phase.
        yellow: Yellow time (s).

    Returns:
        The XML content of the additional file.
    """
    lines = [
        "<additional>",
        f'    <tlLogic id="{spec.tls_id}" type="actuated" '
        f'programID="{ACTUATED_PROGRAM_ID}" offset="0">',
    ]
    for green, yellow_state in zip(spec.green_states, spec.yellow_states, strict=True):
        lines.append(
            f'        <phase duration="{min_green}" minDur="{min_green}" '
            f'maxDur="{max_green}" state="{green}"/>'
        )
        lines.append(f'        <phase duration="{yellow}" state="{yellow_state}"/>')
    lines += ["    </tlLogic>", "</additional>", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class FixedTimeController:
    """Cycles through the actions in order, each with its own green time."""

    def __init__(self, green_durations: Sequence[int]) -> None:
        """Initialize the controller.

        Args:
            green_durations: Green time (s) for each action, in action order.
        """
        if not green_durations:
            msg = "Expected at least one green duration"
            raise ValueError(msg)
        self.green_durations = list(green_durations)
        self._next_action = 0

    def choose_action(self, state: NDArray) -> int:  # noqa: ARG002
        """Return the next action of the cycle."""
        action = self._next_action
        self._next_action = (action + 1) % len(self.green_durations)
        return action

    def green_duration_for(self, action: int) -> int:
        """Return the configured green time for the action."""
        return self.green_durations[action]


class RandomController:
    """Picks a uniformly random action at every decision step (sanity-check lower bound)."""

    def __init__(self, seed: int, num_actions: int) -> None:
        """Initialize the controller with a seeded RNG for reproducibility."""
        self.rng = np.random.default_rng(seed)
        self.num_actions = num_actions

    def choose_action(self, state: NDArray) -> int:  # noqa: ARG002
        """Return a random action."""
        return int(self.rng.integers(self.num_actions))


class LongestQueueController:
    """Gives green to the action serving the longest total queue."""

    def __init__(self, env: Environment) -> None:
        """Initialize the controller with the environment it observes."""
        self.env = env
        self.lanes = all_lanes(env.spec)

    def choose_action(self, state: NDArray) -> int:  # noqa: ARG002
        """Return the action with the most halting vehicles on its incoming lanes."""
        scores = queue_scores(self.env.get_lane_halting_numbers(self.lanes), self.env.spec)
        return argmax_prefer_current(scores, self.env.get_current_action())


class MaxPressureController:
    """Gives green to the action with the highest pressure (upstream minus downstream vehicles)."""

    def __init__(self, env: Environment) -> None:
        """Initialize the controller with the environment it observes."""
        self.env = env
        self.lanes = all_lanes(env.spec)

    def choose_action(self, state: NDArray) -> int:  # noqa: ARG002
        """Return the action with the highest pressure."""
        scores = pressure_scores(self.env.get_lane_vehicle_numbers(self.lanes), self.env.spec)
        return argmax_prefer_current(scores, self.env.get_current_action())


def make_controller(
    name: ControllerName,
    settings: TestingSettings,
    env: Environment,
) -> Policy | None:
    """Build the controller for a baseline run.

    Args:
        name: Which baseline to build.
        settings: Testing settings (scenario and baseline options).
        env: Environment the controller acts on.

    Returns:
        A policy, or None for the actuated baseline (SUMO controls the light).
    """
    match name:
        case ControllerName.FIXED:
            n = env.num_actions
            durations = settings.fixed_green_durations or [settings.green_duration] * n
            if len(durations) != n:
                msg = (
                    f"fixed_green_durations has {len(durations)} values but network "
                    f"'{settings.network}' has {n} green phases (one value per phase)"
                )
                raise ValueError(msg)
            return FixedTimeController(durations)
        case ControllerName.RANDOM:
            return RandomController(seed=settings.episode_seed, num_actions=env.num_actions)
        case ControllerName.LQF:
            return LongestQueueController(env)
        case ControllerName.MAXPRESSURE:
            return MaxPressureController(env)
        case ControllerName.ACTUATED:
            return None


def make_test_environment(
    controller: ControllerName | str,
    settings: TestingSettings,
    tripinfo_file: Path,
    sumocfg_file: Path | None = None,
    cells_per_lane: int = DEFAULT_CELLS_PER_LANE,
) -> Environment:
    """Build the test-scenario environment for a controller.

    For the actuated baseline this also writes the actuated program file and makes the
    environment load and activate it.

    Args:
        controller: Controller name ("dqn" or a baseline).
        settings: Testing settings.
        tripinfo_file: Where SUMO writes per-vehicle trip info.
        sumocfg_file: SUMO config to use instead of the network folder's one.
        cells_per_lane: State cells per incoming lane (only matters for the DQN agent).

    Returns:
        The environment (not yet started).
    """
    spec = load_network_spec(settings.network, settings.tls_id, settings.signal_program)
    additional_files: list[Path] = []
    tl_program = None
    if controller == ControllerName.ACTUATED:
        spec.actuated_program_file.write_text(
            build_actuated_program(
                spec,
                min_green=settings.green_duration,
                max_green=settings.actuated_max_green,
                yellow=settings.yellow_duration,
            ),
            encoding="utf-8",
        )
        additional_files = [spec.actuated_program_file]
        tl_program = ACTUATED_PROGRAM_ID

    return Environment(
        spec=spec,
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        gui=settings.gui,
        cells_per_lane=cells_per_lane,
        sumocfg_file=sumocfg_file,
        tripinfo_file=tripinfo_file,
        additional_files=additional_files,
        tl_program=tl_program,
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def baseline_session(
    controller: ControllerName,
    settings_file: Path,
    out_path: Path,
) -> None:
    """Run one baseline controller on the seeded test scenario and save the results.

    Args:
        controller: Which baseline controller to run.
        settings_file: Path to the testing settings file (same scenario as `tlcs test`).
        out_path: Folder where plots and data are saved.
    """
    settings = load_testing_settings(settings_file)
    out_path.mkdir(parents=True, exist_ok=True)

    env = make_test_environment(controller, settings, tripinfo_file=out_path / TRIPINFO_FILE)
    policy = make_controller(controller, settings, env)
    logger.info(f"Running baseline '{controller}' with episode seed {settings.episode_seed}")

    episode_history, env_stats = run_episode(env=env, agent=policy, seed=settings.episode_seed)

    save_test_results(
        out_path=out_path,
        settings_file=settings_file,
        settings=settings,
        controller=str(controller),
        episode_history=episode_history,
        env_stats=env_stats,
    )
