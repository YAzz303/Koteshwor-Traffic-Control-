"""Network discovery: signal program parsing, state cells, and the two bundled networks."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tlcs.network import (
    NETWORKS_DIR,
    build_network_spec,
    cell_boundaries,
    cell_index,
    greens_and_yellows,
    is_green,
    is_yellow,
    load_network_spec,
    make_yellow,
    transition_yellow,
)

# --- signal program parsing ------------------------------------------------------------------


def test_green_and_yellow_classification() -> None:
    assert is_green("GGrr")
    assert is_green("rgrr")  # permissive green counts
    assert not is_green("yyrr")
    assert not is_green("rrrr")
    assert is_yellow("yyrr")
    assert is_yellow("rrrGyyyyy")  # a link may stay green through the clearance
    assert not is_yellow("GGrr")


def test_greens_and_yellows_use_program_yellows() -> None:
    greens, yellows = greens_and_yellows(["GGrr", "yyrr", "rrGG", "rryy"])
    assert greens == ["GGrr", "rrGG"]
    assert yellows == ["yyrr", "rryy"]


def test_missing_yellows_are_generated() -> None:
    # No yellow between the greens, a duplicate green and an all-red phase.
    greens, yellows = greens_and_yellows(["GGrr", "rrGg", "rrrr", "GGrr"])
    assert greens == ["GGrr", "rrGg"]
    assert yellows == ["yyrr", "rryy"]
    assert make_yellow("GgrG") == "yyry"


def test_transition_yellow_never_jumps_green_to_red() -> None:
    # The program's yellow keeps link 3 green; fine if the next green keeps it too...
    assert transition_yellow("rrrGyyyyy", "GGGGrrrrr") == "rrrGyyyyy"
    # ...but if the next green turns link 3 red, it must go through yellow.
    assert transition_yellow("rrrGyyyyy", "GGGrrrrrr") == "rrryyyyyy"


# --- state cells -----------------------------------------------------------------------------


def test_cell_boundaries_grow_and_cover_the_lane() -> None:
    bounds = cell_boundaries(750.0, 5)
    assert len(bounds) == 5
    assert bounds[0] == pytest.approx(7.5)
    assert bounds[-1] == 750.0
    widths = [b - a for a, b in zip([0.0, *bounds], bounds, strict=False)]
    assert widths == sorted(widths)  # cells get longer away from the stop line


def test_cell_boundaries_short_lane_and_single_cell() -> None:
    assert cell_boundaries(20.0, 4) == [5.0, 10.0, 15.0, 20.0]  # too short: equal cells
    assert cell_boundaries(300.0, 1) == [300.0]


def test_cell_index() -> None:
    bounds = [7.5, 30.0, 100.0]
    assert cell_index(0.0, bounds) == 0
    assert cell_index(7.5, bounds) == 0
    assert cell_index(8.0, bounds) == 1
    assert cell_index(99.0, bounds) == 2
    assert cell_index(500.0, bounds) == 2  # beyond the lane end: last cell


# --- bundled networks ------------------------------------------------------------------------


def test_original_network_is_discovered() -> None:
    spec = load_network_spec("original")
    assert spec.tls_id == "TL"
    assert spec.num_actions == 4
    assert len(spec.incoming_lanes) == 16  # 16 lanes x 5 cells = state size 80
    assert set(spec.incoming_edges) == {"N2TL", "E2TL", "S2TL", "W2TL"}
    assert len(spec.routes) == 12
    assert sum(r.is_straight for r in spec.routes) == 4


def test_t_junction_is_three_arm_left_hand() -> None:
    spec = load_network_spec("t_junction")
    net = ET.parse(spec.net_file).getroot()  # noqa: S314
    assert net.get("lefthand") == "true"
    assert len(spec.incoming_edges) == 3
    assert spec.num_actions == 2
    assert len(spec.incoming_lanes) == 6

    moves = {(r.in_edge, r.out_edge): r.direction for r in spec.routes}
    # Left-hand traffic: the kerb-side turn is LEFT, the crossing turn is RIGHT.
    assert moves == {
        ("W2C", "C2E"): "s", ("E2C", "C2W"): "s",
        ("E2C", "C2S"): "l", ("S2C", "C2W"): "l",
        ("W2C", "C2S"): "r", ("S2C", "C2E"): "r",
    }  # fmt: skip
    # Every green/yellow state has one character per controlled link.
    assert {len(s) for s in spec.green_states + spec.yellow_states} == {9}


def test_tls_id_selection() -> None:
    net_file = NETWORKS_DIR / "t_junction" / "t_junction.net.xml"
    assert build_network_spec(net_file, tls_id="C").tls_id == "C"
    with pytest.raises(ValueError, match="not found"):
        build_network_spec(net_file, tls_id="nope")


def test_missing_network_folder() -> None:
    with pytest.raises(FileNotFoundError, match="Network folder not found"):
        load_network_spec("does_not_exist")


def test_network_folders_have_one_net_and_a_config() -> None:
    for name in ("original", "t_junction"):
        folder = Path(NETWORKS_DIR / name)
        assert len(list(folder.glob("*.net.xml"))) == 1
        assert (folder / "sumo_config.sumocfg").exists()
