from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import traci
from numpy.typing import NDArray
from sumolib import checkBinary

from tlcs.generator import generate_routefile
from tlcs.network import NetworkSpec, cell_boundaries, cell_index, transition_yellow

DEFAULT_CELLS_PER_LANE = 5


@dataclass
class EnvStats:
    """Snapshot of environment statistics for a single simulation step."""

    queue_length: int


# A signal state string (one character per controlled link) and how many steps to hold it.
SignalSegment = tuple[str, int]


def build_phase_plan(  # noqa: PLR0913
    current_action: int,
    action: int,
    green_states: Sequence[str],
    yellow_states: Sequence[str],
    yellow_duration: int,
    green_duration: int,
) -> list[SignalSegment]:
    """Return the signal segments needed to execute an action from the current one.

    If the requested green differs from the current green, the yellow of the current green is
    inserted first. This is the single definition of the yellow-insertion rule, shared by
    `Environment.execute` and the step-by-step live viewer.

    Args:
        current_action: Action whose green is active, or -1 if no green is active.
        action: Requested action index.
        green_states: Green signal state of each action.
        yellow_states: Yellow signal state ending each action's green.
        yellow_duration: Steps of yellow to insert on a change.
        green_duration: Steps of green for the requested action.

    Returns:
        List of (signal state, duration) segments, applied in order.
    """
    plan: list[SignalSegment] = []
    if current_action not in (-1, action):
        yellow = transition_yellow(yellow_states[current_action], green_states[action])
        plan.append((yellow, yellow_duration))
    plan.append((green_states[action], green_duration))
    return plan


class Environment:
    """Reinforcement-learning environment wrapper around a SUMO simulation of one junction."""

    def __init__(  # noqa: PLR0913
        self,
        spec: NetworkSpec,
        n_cars_generated: int,
        max_steps: int,
        yellow_duration: int,
        green_duration: int,
        turn_chance: float,
        gui: bool,
        cells_per_lane: int = DEFAULT_CELLS_PER_LANE,
        tripinfo_file: Path | None = None,
        additional_files: list[Path] | None = None,
        tl_program: str | None = None,
        sumocfg_file: Path | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            spec: The junction to control (see `tlcs.network`).
            n_cars_generated: Number of cars to generate for the episode.
            max_steps: Maximum number of simulation steps in an episode.
            yellow_duration: Number of steps to hold a yellow phase.
            green_duration: Number of steps to hold a green phase.
            turn_chance: Probability for each car to turn instead of going straight.
            gui: Whether to use the SUMO GUI binary.
            cells_per_lane: Number of state cells per incoming lane.
            tripinfo_file: If set, SUMO writes per-vehicle trip info (incl. unfinished trips) here.
            additional_files: Extra SUMO additional files to load (e.g. a traffic-light program).
            tl_program: If set, switch the traffic light to this program ID after start.
            sumocfg_file: SUMO config to use instead of the network folder's one.
        """
        self.spec = spec
        self.n_cars_generated = n_cars_generated
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration
        self.green_duration = green_duration
        self.turn_chance = turn_chance
        self.gui = gui
        self.cells_per_lane = cells_per_lane
        self.tripinfo_file = tripinfo_file
        # The selected signal program (if any) is loaded so SUMO starts in its first phase.
        program = [spec.program_file] if spec.program_file is not None else []
        self.additional_files = program + (additional_files or [])
        self.tl_program = tl_program
        self.sumocfg_file = sumocfg_file or spec.sumocfg_file

        self._lane_bounds = [
            cell_boundaries(length, cells_per_lane) for length in spec.incoming_lane_lengths
        ]
        self._incoming_edges = set(spec.incoming_edges)

        self.step = 0

    @property
    def state_size(self) -> int:
        """Length of the observation vector."""
        return len(self.spec.incoming_lanes) * self.cells_per_lane

    @property
    def num_actions(self) -> int:
        """Number of actions (green phases)."""
        return self.spec.num_actions

    def build_sumo_cmd(self) -> list[str]:
        """Build the SUMO command line based on configuration settings.

        Returns:
            List of command-line arguments to start SUMO.
        """
        sumo_binary = checkBinary("sumo-gui" if self.gui else "sumo")

        if not self.sumocfg_file.exists():
            msg = f"SUMO config not found at '{self.sumocfg_file}'"
            raise FileNotFoundError(msg)

        cmd = [
            sumo_binary,
            "-c",
            str(self.sumocfg_file),
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            str(self.max_steps),
        ]

        if self.tripinfo_file is not None:
            cmd += [
                "--tripinfo-output",
                str(self.tripinfo_file.resolve()),
                "--tripinfo-output.write-unfinished",
                "true",
            ]

        if self.additional_files:
            cmd += ["--additional-files", ",".join(str(p.resolve()) for p in self.additional_files)]

        return cmd

    def activate(self, extra_args: list[str] | None = None, label: str = "default") -> None:
        """Start the SUMO simulation.

        Args:
            extra_args: Additional SUMO command-line arguments (e.g. GUI window placement).
            label: TraCI connection label; use distinct labels to run several simulations at once.
        """
        sumo_cmd = self.build_sumo_cmd() + (extra_args or [])
        traci.start(sumo_cmd, label=label)
        program = self.tl_program or self.spec.program_id
        if program is not None:
            traci.trafficlight.setProgram(self.spec.tls_id, program)

    def deactivate(self) -> None:
        """Stop the SUMO simulation."""
        traci.close()

    def is_over(self) -> bool:
        """Check whether the maximum number of steps has been reached.

        Returns:
            True if the episode is finished, False otherwise.
        """
        return self.step >= self.max_steps

    def generate_routefile(self, seed: int) -> None:
        """Generate a route file for the current episode.

        Args:
            seed: Random seed used for route generation.
        """
        generate_routefile(
            seed=seed,
            n_cars_generated=self.n_cars_generated,
            max_steps=self.max_steps,
            turn_chance=self.turn_chance,
            routes_file=self.spec.routes_file,
            routes=self.spec.routes,
        )

    def get_state(self) -> NDArray:
        """Compute the occupancy state of the incoming lanes.

        Each incoming lane is split into `cells_per_lane` cells (see
        `tlcs.network.cell_boundaries`); a cell is 1 if at least one vehicle is in it.

        Returns:
            A NumPy array of shape (state_size,) with 0/1 occupancy values.
        """
        state = np.zeros(self.state_size, dtype=float)

        for i, (lane, length) in enumerate(
            zip(self.spec.incoming_lanes, self.spec.incoming_lane_lengths, strict=True)
        ):
            bounds = self._lane_bounds[i]
            for car_id in traci.lane.getLastStepVehicleIDs(lane):
                distance_to_stop = length - float(traci.vehicle.getLanePosition(car_id))
                state[i * self.cells_per_lane + cell_index(distance_to_stop, bounds)] = 1.0

        return state

    def get_cumulated_waiting_time(self) -> float:
        """Compute the sum of waiting times for vehicles on incoming edges.

        Returns:
            Total accumulated waiting time of all vehicles on incoming edges.
        """
        waiting_times = 0.0

        for car_id in traci.vehicle.getIDList():
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id not in self._incoming_edges:
                continue
            wait_time = float(traci.vehicle.getAccumulatedWaitingTime(car_id))
            waiting_times += wait_time

        return waiting_times

    def _simulate(self, duration: int) -> list[EnvStats]:
        """Advance the simulation for a given number of steps.

        The actual number of steps is capped so as not to exceed `max_steps`.

        Args:
            duration: Desired number of simulation steps.

        Returns:
            A list of EnvStats, one entry per simulation step.
        """
        stats: list[EnvStats] = []
        steps_todo = min(duration, self.max_steps - self.step)

        for _ in range(steps_todo):
            traci.simulationStep()
            self.step += 1
            queue_length = self.get_queue_length()
            stats.append(EnvStats(queue_length=queue_length))

        return stats

    def advance(self, duration: int) -> list[EnvStats]:
        """Advance the simulation without touching the traffic light.

        Used when SUMO's own signal logic (e.g. an actuated program) controls the light.

        Args:
            duration: Desired number of simulation steps (capped at `max_steps`).

        Returns:
            A list of EnvStats, one entry per simulation step.
        """
        return self._simulate(duration)

    def get_current_action(self) -> int:
        """Return the action whose green is currently shown, or -1 (yellow or other state).

        Returns:
            Action index, or -1 if no action's green is active.
        """
        return self.spec.action_of_state(
            traci.trafficlight.getRedYellowGreenState(self.spec.tls_id)
        )

    def get_lane_halting_numbers(self, lane_ids: Iterable[str]) -> dict[str, int]:
        """Return the number of stopped vehicles on each given lane.

        Args:
            lane_ids: SUMO lane IDs.

        Returns:
            Mapping lane ID -> halting vehicle count in the last step.
        """
        return {lane: int(traci.lane.getLastStepHaltingNumber(lane)) for lane in lane_ids}

    def get_lane_vehicle_numbers(self, lane_ids: Iterable[str]) -> dict[str, int]:
        """Return the number of vehicles (moving or stopped) on each given lane.

        Args:
            lane_ids: SUMO lane IDs.

        Returns:
            Mapping lane ID -> vehicle count in the last step.
        """
        return {lane: int(traci.lane.getLastStepVehicleNumber(lane)) for lane in lane_ids}

    def execute(self, action: int, green_duration: int | None = None) -> list[EnvStats]:
        """Execute an action by showing its green, with a yellow first if the green changes.

        Args:
            action: Action index (one green phase of the discovered signal program).
            green_duration: Green time for this action; defaults to the configured green_duration.

        Returns:
            A list of EnvStats collected during the applied signal states.
        """
        stats: list[EnvStats] = []

        for signal, duration in self.plan_action(action, green_duration):
            if self.is_over():
                break
            self.set_signal(signal)
            stats.extend(self._simulate(duration))

        return stats

    def plan_action(self, action: int, green_duration: int | None = None) -> list[SignalSegment]:
        """Return the (signal state, duration) segments that executing `action` now would apply.

        Args:
            action: Action index.
            green_duration: Green time for this action; defaults to the configured green_duration.

        Returns:
            The signal plan (see `build_phase_plan`).
        """
        return build_phase_plan(
            current_action=self.get_current_action(),
            action=action,
            green_states=self.spec.green_states,
            yellow_states=self.spec.yellow_states,
            yellow_duration=self.yellow_duration,
            green_duration=self.green_duration if green_duration is None else green_duration,
        )

    def set_signal(self, signal: str) -> None:
        """Show the given signal state on the traffic light.

        Args:
            signal: One character per controlled link (e.g. "GGrr").
        """
        traci.trafficlight.setRedYellowGreenState(self.spec.tls_id, signal)

    def get_queue_length(self) -> int:
        """Return the number of stopped vehicles on all incoming edges.

        Returns:
            Total number of vehicles with speed 0 on incoming edges.
        """
        return sum(
            int(traci.edge.getLastStepHaltingNumber(edge)) for edge in self.spec.incoming_edges
        )
