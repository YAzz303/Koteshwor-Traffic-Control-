from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tlcs.network import Route

# Passenger cars only (SUMO's default vehicle class is "passenger").
VEHICLE_TYPE = (
    '    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" '
    'maxSpeed="25" sigma="0.5" />'
)


def _map_to_interval(values: NDArray, new_min: int, new_max: int) -> NDArray:
    """Linearly map values to the interval [new_min, new_max].

    Falls back to a constant array if the input has zero range.

    Args:
        values: Input array of values to be re-scaled.
        new_min: Minimum value of the target interval.
        new_max: Maximum value of the target interval.

    Returns:
        Array of values mapped to the target interval, same shape as `values`.
    """
    old_min = float(values.min())
    old_max = float(values.max())
    return np.interp(values, (old_min, old_max), (new_min, new_max))


def route_id(route: Route) -> str:
    """ID of a route in the generated file, e.g. "W2C_to_C2E"."""
    return f"{route.in_edge}_to_{route.out_edge}"


def split_routes(routes: Sequence[Route]) -> tuple[list[str], list[str]]:
    """Split route IDs into straight and turning movements.

    Args:
        routes: Movements through the junction.

    Returns:
        (straight route IDs, turning route IDs). U-turns get no traffic. If one group is empty,
        it gets the other's routes, so every car always has a route to pick.
    """
    straight = [route_id(r) for r in routes if r.is_straight]
    turning = [route_id(r) for r in routes if not r.is_straight and not r.is_uturn]
    return straight or turning, turning or straight


def build_routes_header(routes: Sequence[Route]) -> str:
    """Opening of the route file: the car type and one route per movement."""
    lines = ["<routes>", VEHICLE_TYPE, ""]
    lines += [f'    <route id="{route_id(r)}" edges="{" ".join(r.path)}"/>' for r in routes]
    return "\n".join(lines)


def _get_car_row(route_id: str, car_i: int, step: int) -> str:
    """Build the XML row describing a single vehicle.

    Args:
        route_id: Identifier of the route the vehicle will follow.
        car_i: Index of the vehicle in the episode.
        step: Simulation step at which the vehicle departs.

    Returns:
        XML snippet representing the vehicle element.
    """
    return f'    <vehicle id="{route_id}_{car_i}" type="standard_car" route="{route_id}" depart="{step}" departLane="random" departSpeed="10" />'  # noqa: E501


def generate_routefile(  # noqa: PLR0913
    seed: int,
    n_cars_generated: int,
    max_steps: int,
    turn_chance: float,
    routes_file: Path,
    routes: Sequence[Route],
) -> None:
    """Generate a SUMO route file for one simulation episode.

    Car departure times follow a Weibull distribution re-scaled to [0, max_steps].
    A fraction of cars go straight and the rest turn, controlled by turn_chance; within each
    group the movement is picked uniformly among the junction's movements.

    Args:
        seed: Random seed for reproducible generation.
        n_cars_generated: Number of cars to generate in the episode.
        max_steps: Maximum simulation step for car departures.
        turn_chance: Probability to select a turn route rather than a straight route.
        routes_file: Where to write the route file.
        routes: Movements through the junction (from the network spec).
    """
    rng = np.random.default_rng(seed)
    straight_routes, turn_routes = split_routes(routes)

    # Generate departure timings according to a Weibull distribution
    timings = np.sort(rng.weibull(2.0, size=n_cars_generated))

    # Rescale the distribution to the interval [0, max_steps]
    generated_steps = _map_to_interval(timings, new_min=0, new_max=max_steps)

    # Round to integer steps -> effective depart times for each car
    depart_steps = np.rint(generated_steps).astype(int)

    routes_file.parent.mkdir(parents=True, exist_ok=True)

    with routes_file.open("w", encoding="utf-8") as out:
        print(build_routes_header(routes), file=out)

        for car_i, step in enumerate(depart_steps):
            routes_selected = turn_routes if rng.random() < turn_chance else straight_routes
            chosen = rng.choice(routes_selected)
            car_row = _get_car_row(route_id=chosen, car_i=car_i, step=step)

            print(car_row, file=out)

        print("</routes>", file=out)
