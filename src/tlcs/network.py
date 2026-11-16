"""Discover a signalized junction from a SUMO network file.

Everything the environment, agent and baselines need to know about the junction (traffic light,
incoming/outgoing lanes, movements, green and yellow signal states) is read from the
`*.net.xml` file with sumolib, so any single signalized junction works: any number of arms,
lanes or phases, left- or right-hand traffic. No SUMO process is needed to build the spec.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import sumolib

NETWORKS_DIR = Path("intersection")
SUMOCFG_NAME = "sumo_config.sumocfg"
ROUTES_NAME = "episode_routes.rou.xml"
ACTUATED_NAME = "tls_actuated.add.xml"

GREEN_CHARS = "Gg"
YELLOW_CHARS = "yY"

Movement = tuple[str, str]  # (incoming lane ID, outgoing lane ID)


@dataclass(frozen=True)
class Route:
    """A movement through the junction at road level, used to generate traffic."""

    in_edge: str
    out_edge: str
    direction: str  # SUMO link direction: "s" straight, "l"/"r" left/right, "L"/"R" partial
    # Full edge path the cars drive: from the network edge, through the junction, to the edge.
    # Defaults to just (in_edge, out_edge).
    edges: tuple[str, ...] = ()
    # True if the car leaves the junction back the way it came (turns by more than
    # UTURN_MIN_TURN degrees), whatever SUMO's direction label says.
    uturn: bool = False

    @property
    def path(self) -> tuple[str, ...]:
        """Edges of the route, from entry to exit."""
        return self.edges or (self.in_edge, self.out_edge)

    @property
    def is_straight(self) -> bool:
        """True for straight-through movements."""
        return self.direction == "s"

    @property
    def is_uturn(self) -> bool:
        """True for U-turns (by geometry, or SUMO direction "t"/"T")."""
        return self.uturn or self.direction in "tT"


@dataclass(frozen=True)
class NetworkSpec:
    """Everything the RL code needs to know about one signalized junction."""

    directory: Path
    net_file: Path
    tls_id: str
    incoming_lanes: tuple[str, ...]
    incoming_lane_lengths: tuple[float, ...]
    incoming_edges: tuple[str, ...]
    outgoing_lanes: tuple[str, ...]
    green_states: tuple[str, ...]
    yellow_states: tuple[str, ...]
    action_movements: tuple[tuple[Movement, ...], ...]
    routes: tuple[Route, ...]
    center: tuple[float, float]
    # Signal program file (in the network folder) the phases come from; None = the net file's.
    program_file: Path | None = None
    program_id: str | None = None

    @property
    def num_actions(self) -> int:
        """Number of actions (green phases)."""
        return len(self.green_states)

    @property
    def sumocfg_file(self) -> Path:
        """SUMO config of the network folder."""
        return self.directory / SUMOCFG_NAME

    @property
    def routes_file(self) -> Path:
        """Route file generated for each episode (referenced by the SUMO config)."""
        return self.directory / ROUTES_NAME

    @property
    def actuated_program_file(self) -> Path:
        """Generated additional file with the actuated program (actuated baseline)."""
        return self.directory / ACTUATED_NAME

    def action_lanes(self, action: int) -> tuple[list[str], list[str]]:
        """Return the distinct incoming and outgoing lanes served by an action's green.

        Args:
            action: Action index.

        Returns:
            (incoming lanes, outgoing lanes), without duplicates, in link order.
        """
        movements = self.action_movements[action]
        in_lanes = list(dict.fromkeys(src for src, _ in movements))
        out_lanes = list(dict.fromkeys(dst for _, dst in movements))
        return in_lanes, out_lanes

    def action_of_state(self, state: str) -> int:
        """Return the action whose green signal state equals `state`, or -1 (e.g. yellow)."""
        try:
            return self.green_states.index(state)
        except ValueError:
            return -1


def is_green(state: str) -> bool:
    """A green phase gives green to at least one link and shows no yellow."""
    return any(c in GREEN_CHARS for c in state) and not any(c in YELLOW_CHARS for c in state)


def is_yellow(state: str) -> bool:
    """A yellow (clearance) phase shows yellow on at least one link.

    Links that stay green through the clearance (e.g. "rrrGyyyyy") are allowed.
    """
    return any(c in YELLOW_CHARS for c in state)


def make_yellow(green_state: str) -> str:
    """Generate the yellow that ends a green: every green link turns yellow, the rest is kept."""
    return "".join("y" if c in GREEN_CHARS else c for c in green_state)


def transition_yellow(yellow_state: str, next_green: str) -> str:
    """Adapt a green's yellow to the green that follows it.

    A link the yellow keeps green must also be green next; otherwise it turns yellow, so no link
    ever jumps from green straight to red. (The agent can pick any next green, not only the next
    one in the signal program the yellow was written for.)

    Args:
        yellow_state: Yellow that ends the current green.
        next_green: Green state that will follow.

    Returns:
        The yellow state to show.
    """
    return "".join(
        "y" if c in GREEN_CHARS and n not in GREEN_CHARS else c
        for c, n in zip(yellow_state, next_green, strict=True)
    )


def greens_and_yellows(phase_states: list[str]) -> tuple[list[str], list[str]]:
    """Derive the actions (green states) and their yellows from a signal program.

    Each distinct green phase becomes one action, in program order. Its yellow is the phase that
    follows it in the program if that phase is a pure yellow; otherwise a yellow is generated.

    Args:
        phase_states: Signal state of every phase of the program, in order.

    Returns:
        (green states, yellow states), one yellow per green.
    """
    greens: list[str] = []
    yellows: list[str] = []
    n = len(phase_states)
    for i, state in enumerate(phase_states):
        if not is_green(state) or state in greens:
            continue
        following = phase_states[(i + 1) % n]
        greens.append(state)
        yellows.append(following if is_yellow(following) else make_yellow(state))
    return greens, yellows


def find_net_file(directory: Path) -> Path:
    """Return the single `*.net.xml` file in a network folder."""
    candidates = sorted(directory.glob("*.net.xml"))
    if len(candidates) != 1:
        msg = f"Expected exactly one *.net.xml in '{directory}', found {len(candidates)}"
        raise FileNotFoundError(msg)
    return candidates[0]


def load_network_spec(
    network: str,
    tls_id: str | None = None,
    signal_program: str | None = None,
) -> NetworkSpec:
    """Read the junction of a network folder under `intersection/`.

    Args:
        network: Folder name under `intersection/` (e.g. "original").
        tls_id: Traffic light to control; None means the only one in the network.
        signal_program: tlLogic file in the network folder whose phases define the actions;
            None uses the program in the net file.

    Returns:
        The discovered NetworkSpec.
    """
    directory = NETWORKS_DIR / network
    if not directory.is_dir():
        msg = f"Network folder not found: '{directory}'"
        raise FileNotFoundError(msg)
    program_file = None
    if signal_program is not None:
        program_file = directory / signal_program
        if not program_file.exists():
            msg = f"Signal program file not found: '{program_file}'"
            raise FileNotFoundError(msg)
    return build_network_spec(
        find_net_file(directory), tls_id=tls_id, directory=directory, program_file=program_file
    )


def read_program_file(program_file: Path, tls_id: str) -> tuple[str, list[str]]:
    """Return (programID, phase states) of the tlLogic for `tls_id` in an additional file."""
    root = ET.parse(program_file).getroot()  # noqa: S314
    for logic in root.iter("tlLogic"):
        if logic.get("id") == tls_id:
            states = [phase.get("state", "") for phase in logic.iter("phase")]
            return logic.get("programID", ""), states
    msg = f"No tlLogic for traffic light '{tls_id}' in '{program_file}'"
    raise ValueError(msg)


# Edges continue a road if they turn by at most this much (degrees); used to extend routes.
MAX_CONTINUATION_TURN = 60.0
# A movement turning by more than this (degrees) goes back the way it came: a U-turn.
UTURN_MIN_TURN = 135.0


def turn_angle(in_edge: sumolib.net.edge.Edge, out_edge: sumolib.net.edge.Edge) -> float:
    """Absolute heading change (degrees) from the end of in_edge to the start of out_edge."""
    a, b = in_edge.getShape()[-2:], out_edge.getShape()[:2]
    return abs((_heading(*b) - _heading(*a) + 180) % 360 - 180)


def _heading(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.degrees(math.atan2(q[0] - p[0], q[1] - p[1])) % 360


def follow_road(edge: sumolib.net.edge.Edge, forward: bool, avoid: set[str]) -> list[str]:
    """Follow the road from `edge` (downstream if forward, else upstream) to the network edge.

    At each step the connected edge with the smallest turn is taken, as long as the turn is at
    most MAX_CONTINUATION_TURN degrees and the edge does not touch the controlled junction.
    """
    path: list[str] = []
    seen = {edge.getID()}
    current = edge
    while True:
        candidates = current.getOutgoing() if forward else current.getIncoming()
        shape = current.getShape()
        here = _heading(shape[-2], shape[-1]) if forward else _heading(shape[0], shape[1])
        best, best_turn = None, MAX_CONTINUATION_TURN
        for cand in candidates:
            if cand.getID() in seen or cand.getID() in avoid:
                continue
            s = cand.getShape()
            there = _heading(s[0], s[1]) if forward else _heading(s[-2], s[-1])
            turn = abs((there - here + 180) % 360 - 180)
            if turn <= best_turn:
                best, best_turn = cand, turn
        if best is None:
            return path
        path.append(best.getID())
        seen.add(best.getID())
        current = best


def build_network_spec(
    net_file: Path,
    tls_id: str | None = None,
    directory: Path | None = None,
    program_file: Path | None = None,
) -> NetworkSpec:
    """Read the controlled junction of a SUMO network file.

    Args:
        net_file: Path to a `*.net.xml` file.
        tls_id: Traffic light to control; None means the only one in the network.
        directory: Network folder (defaults to the net file's folder).
        program_file: tlLogic file whose phases define the actions (None = net file's program).

    Returns:
        The discovered NetworkSpec.
    """
    net = sumolib.net.readNet(str(net_file), withPrograms=True)
    lights = {tls.getID(): tls for tls in net.getTrafficLights()}

    if tls_id is None:
        if len(lights) != 1:
            msg = (
                f"'{net_file}' has {len(lights)} traffic lights {sorted(lights)}; "
                "set `tls_id` to choose one"
            )
            raise ValueError(msg)
        tls_id = next(iter(lights))
    if tls_id not in lights:
        msg = f"Traffic light '{tls_id}' not found in '{net_file}' (has {sorted(lights)})"
        raise ValueError(msg)
    tls = lights[tls_id]

    # Controlled links, ordered by link index (= position in the signal state string).
    links = sorted(tls.getConnections(), key=lambda link: link[2])

    program_id = None
    if program_file is not None:
        program_id, phase_states = read_program_file(program_file, tls_id)
    else:
        programs = tls.getPrograms()
        if not programs:
            msg = f"Traffic light '{tls_id}' has no signal program in '{net_file}'"
            raise ValueError(msg)
        phase_states = [phase.state for phase in programs[sorted(programs)[0]].getPhases()]
    n_links = len(links)
    if any(len(state) != n_links for state in phase_states):
        msg = f"Signal states of '{tls_id}' must have {n_links} characters (one per link)"
        raise ValueError(msg)
    greens, yellows = greens_and_yellows(phase_states)
    if not greens:
        msg = f"Traffic light '{tls_id}' has no green phase in '{net_file}'"
        raise ValueError(msg)

    movements = tuple(
        tuple(
            (in_lane.getID(), out_lane.getID())
            for in_lane, out_lane, index in links
            if green[index] in GREEN_CHARS
        )
        for green in greens
    )

    junction_edges = {lane.getEdge().getID() for link in links for lane in link[:2]}
    routes: dict[tuple[str, str], Route] = {}
    for in_lane, out_lane, _ in links:
        in_edge, out_edge = in_lane.getEdge(), out_lane.getEdge()
        key = (in_edge.getID(), out_edge.getID())
        if key not in routes:
            direction = next(
                c.getDirection() for c in in_lane.getOutgoing() if c.getToLane() is out_lane
            )
            upstream = follow_road(in_edge, forward=False, avoid=junction_edges)
            downstream = follow_road(out_edge, forward=True, avoid=junction_edges)
            routes[key] = Route(
                in_edge=key[0],
                out_edge=key[1],
                direction=direction,
                edges=(*reversed(upstream), key[0], key[1], *downstream),
                uturn=turn_angle(in_edge, out_edge) > UTURN_MIN_TURN,
            )

    in_lanes = list({link[0].getID(): link[0] for link in links}.values())
    out_lanes = list({link[1].getID(): link[1] for link in links}.values())
    stop_points = [lane.getShape()[-1] for lane in in_lanes]

    return NetworkSpec(
        directory=directory or net_file.parent,
        net_file=net_file,
        tls_id=tls_id,
        incoming_lanes=tuple(lane.getID() for lane in in_lanes),
        incoming_lane_lengths=tuple(float(lane.getLength()) for lane in in_lanes),
        incoming_edges=tuple(dict.fromkeys(lane.getEdge().getID() for lane in in_lanes)),
        outgoing_lanes=tuple(lane.getID() for lane in out_lanes),
        green_states=tuple(greens),
        yellow_states=tuple(yellows),
        action_movements=movements,
        routes=tuple(routes.values()),
        center=(
            sum(p[0] for p in stop_points) / len(stop_points),
            sum(p[1] for p in stop_points) / len(stop_points),
        ),
        program_file=program_file,
        program_id=program_id,
    )


# ---------------------------------------------------------------------------
# State discretization
# ---------------------------------------------------------------------------

# Length of the first cell next to the stop line: about one car (5 m) plus its gap (2.5 m).
FIRST_CELL_LENGTH = 7.5


def cell_boundaries(lane_length: float, n_cells: int) -> list[float]:
    """Upper distance-to-stop-line bound of each cell of a lane.

    Cells grow geometrically: the first one is about one car long, the last one ends at the far
    end of the lane. This keeps detail where queues form while still covering the whole lane.
    If the lane is too short for that, equal cells are used.

    Args:
        lane_length: Lane length in meters.
        n_cells: Number of cells.

    Returns:
        n_cells increasing bounds; the last one equals lane_length.
    """
    if n_cells == 1:
        return [lane_length]
    if lane_length <= FIRST_CELL_LENGTH * n_cells:
        return [lane_length * (k + 1) / n_cells for k in range(n_cells)]

    def total(ratio: float) -> float:
        return FIRST_CELL_LENGTH * (ratio**n_cells - 1) / (ratio - 1)

    # Find the growth ratio so the n cells add up to the lane length (bisection).
    lo, hi = 1.0 + 1e-9, 2.0
    while total(hi) < lane_length:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if total(mid) < lane_length else (lo, mid)
    ratio = (lo + hi) / 2

    bounds = [FIRST_CELL_LENGTH * (ratio ** (k + 1) - 1) / (ratio - 1) for k in range(n_cells)]
    bounds[-1] = lane_length
    return bounds


def cell_index(distance_to_stop: float, bounds: list[float]) -> int:
    """Return the cell containing a vehicle at the given distance from the stop line."""
    for i, bound in enumerate(bounds):
        if distance_to_stop <= bound:
            return i
    return len(bounds) - 1
