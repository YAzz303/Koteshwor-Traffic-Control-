import xml.etree.ElementTree as ET

import numpy as np
import pytest

from tlcs.baselines import (
    ControllerName,
    FixedTimeController,
    LongestQueueController,
    MaxPressureController,
    RandomController,
    all_lanes,
    argmax_prefer_current,
    build_actuated_program,
    make_controller,
    pressure_scores,
    queue_scores,
)
from tlcs.constants import ACTUATED_PROGRAM_ID
from tlcs.network import NetworkSpec, load_network_spec
from tlcs.settings import TestingSettings as Settings

STATE = np.zeros(80)

# The hand-written tables of the original intersection, kept here as the regression reference
# for what the network discovery must find.
ORIGINAL_MOVEMENTS = {
    0: {  # NS through + right
        ("N2TL_0", "TL2W_0"), ("N2TL_0", "TL2S_0"), ("N2TL_1", "TL2S_1"), ("N2TL_2", "TL2S_2"),
        ("S2TL_0", "TL2E_0"), ("S2TL_0", "TL2N_0"), ("S2TL_1", "TL2N_1"), ("S2TL_2", "TL2N_2"),
    },
    1: {("N2TL_3", "TL2E_3"), ("S2TL_3", "TL2W_3")},  # NS left
    2: {  # EW through + right
        ("E2TL_0", "TL2N_0"), ("E2TL_0", "TL2W_0"), ("E2TL_1", "TL2W_1"), ("E2TL_2", "TL2W_2"),
        ("W2TL_0", "TL2S_0"), ("W2TL_0", "TL2E_0"), ("W2TL_1", "TL2E_1"), ("W2TL_2", "TL2E_2"),
    },
    3: {("E2TL_3", "TL2S_3"), ("W2TL_3", "TL2N_3")},  # EW left
}  # fmt: skip
ORIGINAL_PROGRAM = (
    "GGGGrrrrrrGGGGrrrrrr", "yyyyrrrrrryyyyrrrrrr", "rrrrGrrrrrrrrrGrrrrr", "rrrryrrrrrrrrryrrrrr",
    "rrrrrGGGGrrrrrrGGGGr", "rrrrryyyyrrrrrryyyyr", "rrrrrrrrrGrrrrrrrrrG", "rrrrrrrrryrrrrrrrrry",
)  # fmt: skip


@pytest.fixture(scope="module")
def original() -> NetworkSpec:
    return load_network_spec("original")


@pytest.fixture(scope="module")
def t_junction() -> NetworkSpec:
    return load_network_spec("t_junction")


# --- discovered movements match the original hand-written tables ----------------------------


def test_original_movements_match_reference(original: NetworkSpec) -> None:
    assert original.num_actions == 4
    for action, expected in ORIGINAL_MOVEMENTS.items():
        assert set(original.action_movements[action]) == expected, f"action {action}"


def test_action_lanes_are_distinct(original: NetworkSpec) -> None:
    in_lanes, out_lanes = original.action_lanes(0)
    assert in_lanes == ["N2TL_0", "N2TL_1", "N2TL_2", "S2TL_0", "S2TL_1", "S2TL_2"]
    assert len(out_lanes) == len(set(out_lanes)) == 8


def test_all_lanes_covers_every_movement(t_junction: NetworkSpec) -> None:
    lanes = set(all_lanes(t_junction))
    for movements in t_junction.action_movements:
        for in_lane, out_lane in movements:
            assert {in_lane, out_lane} <= lanes


# --- pure decision logic ------------------------------------------------------------------


def test_queue_scores_sum_incoming_halting_vehicles(original: NetworkSpec) -> None:
    halting = {"N2TL_3": 4, "S2TL_3": 2, "E2TL_0": 1, "TL2E_3": 99}  # outgoing lanes ignored
    assert queue_scores(halting, original) == [0, 6, 1, 0]


def test_pressure_scores_subtract_outgoing_vehicles(original: NetworkSpec) -> None:
    vehicles = {"N2TL_3": 5, "S2TL_3": 1, "TL2E_3": 2, "W2TL_3": 3, "TL2N_3": 4}
    # action 1 (NS left): (5 + 1) - 2 ; action 3 (EW left): 3 - 4
    assert pressure_scores(vehicles, original) == [0, 4, 0, -1]


def test_scores_have_one_value_per_action_on_t_junction(t_junction: NetworkSpec) -> None:
    counts = dict.fromkeys(all_lanes(t_junction), 1)
    assert len(queue_scores(counts, t_junction)) == 2
    assert len(pressure_scores(counts, t_junction)) == 2


@pytest.mark.parametrize(
    ("scores", "current", "expected"),
    [
        ([1, 5, 2, 0], 0, 1),  # clear winner
        ([3, 1, 3, 0], 2, 2),  # tie including current -> keep current
        ([3, 1, 3, 0], 1, 0),  # tie not including current -> lowest index
        ([0, 0, 0, 0], -1, 0),  # unknown current (yellow) -> lowest index
    ],
)
def test_argmax_prefer_current(scores: list[int], current: int, expected: int) -> None:
    assert argmax_prefer_current(scores, current) == expected


# --- controllers -----------------------------------------------------------------------------


def test_fixed_time_cycles_in_order_with_per_phase_green() -> None:
    controller = FixedTimeController([30, 10, 25, 15])
    actions = [controller.choose_action(STATE) for _ in range(6)]
    assert actions == [0, 1, 2, 3, 0, 1]
    assert [controller.green_duration_for(a) for a in range(4)] == [30, 10, 25, 15]


def test_fixed_time_cycles_any_number_of_phases() -> None:
    controller = FixedTimeController([20, 40])
    assert [controller.choose_action(STATE) for _ in range(3)] == [0, 1, 0]


def test_random_controller_is_seeded_and_in_range() -> None:
    first, second = RandomController(seed=7, num_actions=3), RandomController(seed=7, num_actions=3)
    seq1 = [first.choose_action(STATE) for _ in range(200)]
    seq2 = [second.choose_action(STATE) for _ in range(200)]
    assert seq1 == seq2
    assert set(seq1) == {0, 1, 2}


class FakeEnv:
    """Stands in for Environment: returns fixed lane counts and current action."""

    def __init__(self, spec: NetworkSpec, counts: dict[str, int], current_action: int) -> None:
        self.spec = spec
        self.num_actions = spec.num_actions
        self.counts = counts
        self.current_action = current_action

    def get_lane_halting_numbers(self, lanes: list[str]) -> dict[str, int]:
        return {lane: self.counts.get(lane, 0) for lane in lanes}

    def get_lane_vehicle_numbers(self, lanes: list[str]) -> dict[str, int]:
        return {lane: self.counts.get(lane, 0) for lane in lanes}

    def get_current_action(self) -> int:
        return self.current_action


def test_lqf_controller_picks_longest_queue(original: NetworkSpec) -> None:
    env = FakeEnv(original, {"E2TL_3": 6, "W2TL_3": 1, "N2TL_0": 5}, current_action=0)
    assert LongestQueueController(env).choose_action(STATE) == 3  # type: ignore[arg-type]


def test_maxpressure_controller_accounts_for_downstream(original: NetworkSpec) -> None:
    # EW-left has the most upstream vehicles but its exits are full -> NS through wins.
    env = FakeEnv(original, {"E2TL_3": 6, "TL2S_3": 6, "N2TL_0": 3}, current_action=1)
    assert MaxPressureController(env).choose_action(STATE) == 0  # type: ignore[arg-type]


def _settings(**overrides: object) -> Settings:
    base = {
        "gui": False,
        "max_steps": 100,
        "n_cars_generated": 10,
        "episode_seed": 1,
        "yellow_duration": 4,
        "green_duration": 10,
        "turn_chance": 0.25,
        "gamma": 0.75,
    }
    return Settings.model_validate(base | overrides)


def test_make_controller(original: NetworkSpec, t_junction: NetworkSpec) -> None:
    fixed = make_controller(ControllerName.FIXED, _settings(), FakeEnv(original, {}, 0))  # type: ignore[arg-type]
    assert isinstance(fixed, FixedTimeController)
    assert fixed.green_durations == [10, 10, 10, 10]  # one green_duration per phase

    fixed = make_controller(ControllerName.FIXED, _settings(), FakeEnv(t_junction, {}, 0))  # type: ignore[arg-type]
    assert isinstance(fixed, FixedTimeController)
    assert fixed.green_durations == [10, 10]

    random = make_controller(ControllerName.RANDOM, _settings(), FakeEnv(t_junction, {}, 0))  # type: ignore[arg-type]
    assert isinstance(random, RandomController)
    assert random.num_actions == 2

    assert make_controller(ControllerName.ACTUATED, _settings(), FakeEnv(original, {}, 0)) is None  # type: ignore[arg-type]


def test_fixed_green_durations_must_match_phase_count(t_junction: NetworkSpec) -> None:
    settings = _settings(fixed_green_durations=[30, 10, 30, 10])
    with pytest.raises(ValueError, match="2 green phases"):
        make_controller(ControllerName.FIXED, settings, FakeEnv(t_junction, {}, 0))  # type: ignore[arg-type]


# --- actuated program ------------------------------------------------------------------------


def _phases(xml: str) -> list[ET.Element]:
    tl = ET.fromstring(xml).find("tlLogic")  # noqa: S314
    assert tl is not None
    assert tl.get("type") == "actuated"
    assert tl.get("programID") == ACTUATED_PROGRAM_ID
    return list(tl.iter("phase"))


def test_actuated_program_reproduces_original_program(original: NetworkSpec) -> None:
    phases = _phases(build_actuated_program(original, min_green=10, max_green=45, yellow=4))
    assert tuple(p.get("state") for p in phases) == ORIGINAL_PROGRAM
    for i, phase in enumerate(phases):
        if i % 2 == 0:  # green
            assert (phase.get("minDur"), phase.get("maxDur")) == ("10", "45")
        else:  # yellow, same length as the agent's
            assert phase.get("duration") == "4"
            assert phase.get("minDur") is None


def test_actuated_program_for_t_junction(t_junction: NetworkSpec) -> None:
    phases = _phases(build_actuated_program(t_junction, min_green=10, max_green=45, yellow=4))
    states = [p.get("state") for p in phases]
    assert states == [
        t_junction.green_states[0], t_junction.yellow_states[0],
        t_junction.green_states[1], t_junction.yellow_states[1],
    ]  # fmt: skip
