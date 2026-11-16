from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from numpy.typing import NDArray

from tlcs.env import Environment, EnvStats


@dataclass
class Record:
    """Single time-step transition experienced during an episode.

    Attributes:
        state: Environment state observed before taking the action.
        action: Action chosen by the agent (-1 if SUMO controlled the light itself).
        reward: Reward obtained after executing the action.
        step: Simulation step at which the reward was measured.
    """

    state: NDArray
    action: int
    reward: float
    step: int = 0


class Policy(Protocol):
    """Anything that maps a state to an action (the DQN agent or a baseline controller)."""

    def choose_action(self, state: NDArray) -> int:
        """Return the action to apply in the given state."""
        ...


@runtime_checkable
class VariableGreenPolicy(Protocol):
    """A policy that chooses its own green time per action (e.g. fixed-time control)."""

    def green_duration_for(self, action: int) -> int:
        """Return the green time (s) to apply for the given action."""
        ...


def run_episode(
    env: Environment,
    agent: Policy | None,
    seed: int,
) -> tuple[list[Record], list[EnvStats]]:
    """Runs one episode and returns per-step records and environment statistics.

    Args:
        env: Environment to interact with.
        agent: Policy used to select actions from states. If None, the light is left to SUMO's
            own program (e.g. actuated) and the episode advances in `green_duration` chunks so
            rewards are still measured on a comparable cadence.
        seed: Seed used to generate the route file for this episode.

    Returns:
        A tuple (history, env_stats) where:
            history is the list of per-step records.
            env_stats is the list of environment statistics for each executed action.
    """
    env.generate_routefile(seed=seed)

    previous_total_wait = 0.0
    history: list[Record] = []
    env_stats: list[EnvStats] = []

    env.activate()

    while not env.is_over():
        state = env.get_state()

        if agent is None:
            action = env.get_current_action()
            action_stats = env.advance(env.green_duration)
        else:
            action = agent.choose_action(state)
            green = (
                agent.green_duration_for(action) if isinstance(agent, VariableGreenPolicy) else None
            )
            action_stats = env.execute(action, green_duration=green)
        env_stats.extend(action_stats)

        current_total_wait = env.get_cumulated_waiting_time()
        reward = previous_total_wait - current_total_wait
        previous_total_wait = current_total_wait

        record = Record(state=state, action=action, reward=reward, step=env.step)
        history.append(record)

    env.deactivate()

    return history, env_stats
