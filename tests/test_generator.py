import xml.etree.ElementTree as ET
from pathlib import Path

from tlcs.generator import build_routes_header, generate_routefile, split_routes
from tlcs.network import Route, load_network_spec

T_ROUTES = (
    Route("W2C", "C2E", "s"),
    Route("W2C", "C2S", "r"),
    Route("S2C", "C2W", "l"),
)


def test_split_routes() -> None:
    assert split_routes(T_ROUTES) == (["W2C_to_C2E"], ["W2C_to_C2S", "S2C_to_C2W"])
    # No straight movement at all: every car turns.
    turns_only = (Route("S2C", "C2W", "l"), Route("S2C", "C2E", "r"))
    assert split_routes(turns_only) == (["S2C_to_C2W", "S2C_to_C2E"], ["S2C_to_C2W", "S2C_to_C2E"])


def test_header_declares_one_route_per_movement() -> None:
    header = build_routes_header(T_ROUTES) + "\n</routes>"
    root = ET.fromstring(header)  # noqa: S314
    assert [r.get("edges") for r in root.iter("route")] == ["W2C C2E", "W2C C2S", "S2C C2W"]
    assert [v.get("id") for v in root.iter("vType")] == ["standard_car"]


def _generate(tmp_path: Path, seed: int, turn_chance: float) -> ET.Element:
    out = tmp_path / f"r{seed}_{turn_chance}.rou.xml"
    spec = load_network_spec("t_junction")
    generate_routefile(
        seed=seed,
        n_cars_generated=200,
        max_steps=1000,
        turn_chance=turn_chance,
        routes_file=out,
        routes=spec.routes,
    )
    return ET.parse(out).getroot()  # noqa: S314


def test_generated_file_is_valid_and_reproducible(tmp_path: Path) -> None:
    root = _generate(tmp_path, seed=3, turn_chance=0.25)
    vehicles = list(root.iter("vehicle"))
    assert len(vehicles) == 200
    route_ids = {r.get("id") for r in root.iter("route")}
    assert {v.get("route") for v in vehicles} <= route_ids
    departs = [int(v.get("depart", "-1")) for v in vehicles]
    assert departs == sorted(departs)  # SUMO needs sorted departures
    assert 0 <= departs[0] and departs[-1] <= 1000

    again = _generate(tmp_path, seed=3, turn_chance=0.25)
    assert [v.attrib for v in again.iter("vehicle")] == [v.attrib for v in vehicles]


def test_turn_chance_zero_sends_everyone_straight(tmp_path: Path) -> None:
    root = _generate(tmp_path, seed=1, turn_chance=0.0)
    assert {v.get("route") for v in root.iter("vehicle")} <= {"W2C_to_C2E", "E2C_to_C2W"}
