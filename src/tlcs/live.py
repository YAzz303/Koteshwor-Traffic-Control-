"""Watch several controllers side by side: one SUMO-GUI window per method, stepped in lockstep.

Every method gets its own SUMO process on the same network and the same seeded route file, so
all windows show identical traffic and the same simulation second at the same time. Each
simulation is independent, so the saved results match `tlcs test` / `tlcs baseline` runs.
"""

import math
import sys
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import traci

from tlcs.agent import Agent
from tlcs.baselines import ControllerName, make_controller, make_test_environment
from tlcs.compare import compare_session
from tlcs.constants import MODEL_FILE, TRIPINFO_FILE
from tlcs.env import Environment, EnvStats
from tlcs.episode import Policy, Record, VariableGreenPolicy
from tlcs.generator import generate_routefile
from tlcs.logger import get_logger
from tlcs.main import load_model_settings, save_test_results
from tlcs.network import load_network_spec
from tlcs.settings import TestingSettings, load_testing_settings

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

DQN = "dqn"
ALL_CONTROLLERS = (DQN, *(c.value for c in ControllerName))
LIVE_CFG_PREFIX = "live_"


class StepRunner:
    """Drives one environment one simulation step at a time.

    Produces exactly the same decisions, phases, rewards and statistics as `run_episode`,
    but lets several simulations advance together second by second.
    """

    def __init__(self, label: str, env: Environment, policy: Policy | None) -> None:
        """Initialize the runner.

        Args:
            label: Controller name; also the TraCI connection label.
            env: Started environment for this controller.
            policy: Controller, or None when SUMO's own program controls the light.
        """
        self.label = label
        self.env = env
        self.policy = policy
        self.history: list[Record] = []
        self.env_stats: list[EnvStats] = []
        self._plan: deque[tuple[str | None, int]] = deque()
        self._remaining = 0
        self._previous_wait = 0.0
        self._pending: tuple[NDArray, int] | None = None

    def _close_decision(self) -> None:
        """Record the reward of the decision whose phases just finished."""
        if self._pending is None:
            return
        current_wait = self.env.get_cumulated_waiting_time()
        reward = self._previous_wait - current_wait
        self._previous_wait = current_wait
        state, action = self._pending
        self.history.append(Record(state=state, action=action, reward=reward, step=self.env.step))
        self._pending = None

    def _decide(self) -> None:
        """Observe the state, pick an action and queue its phase plan."""
        state = self.env.get_state()
        if self.policy is None:
            action = self.env.get_current_action()
            plan: list[tuple[str | None, int]] = [(None, self.env.green_duration)]
        else:
            action = self.policy.choose_action(state)
            green = (
                self.policy.green_duration_for(action)
                if isinstance(self.policy, VariableGreenPolicy)
                else None
            )
            plan = list(self.env.plan_action(action, green))
        self._pending = (state, action)
        self._plan = deque(plan)

    def step(self) -> None:
        """Advance this simulation by one step (no-op once the episode is over)."""
        if self.env.is_over():
            return
        while self._remaining == 0:
            if not self._plan:
                self._close_decision()
                self._decide()
            signal, duration = self._plan.popleft()
            if signal is not None:
                self.env.set_signal(signal)
            self._remaining = duration
        self.env_stats.extend(self.env.advance(1))
        self._remaining -= 1

    def finish(self) -> None:
        """Record the reward of the last decision."""
        self._close_decision()


def tile_windows(
    n: int,
    screen_w: int,
    screen_h: int,
    title_bar: int = 40,
) -> list[tuple[int, int, int, int]]:
    """Split the screen into a grid with one window per controller.

    sumo-gui places the window *content* at the given position, with the title bar above it,
    so each cell is shifted down and shortened by `title_bar` to keep every title visible.

    Args:
        n: Number of windows.
        screen_w: Screen width in pixels.
        screen_h: Usable screen height in pixels.
        title_bar: Height in pixels reserved for each window's title bar.

    Returns:
        One (x, y, width, height) tuple per window, row by row.
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    w, h = screen_w // cols, screen_h // rows
    return [((i % cols) * w, (i // cols) * h + title_bar, w, h - title_bar) for i in range(n)]


def screen_size() -> tuple[int, int]:
    """Return the usable screen size (Windows API when available, else 1920x1080)."""
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415

        user32 = ctypes.windll.user32
        taskbar = 48
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1)) - taskbar
    return 1920, 1080


def parse_controllers(value: str) -> list[str]:
    """Parse a comma-separated controller list and validate the names.

    Args:
        value: e.g. "dqn,fixed,lqf".

    Returns:
        Controller names in the given order, without duplicates.
    """
    names = list(dict.fromkeys(v.strip().lower() for v in value.split(",") if v.strip()))
    unknown = [n for n in names if n not in ALL_CONTROLLERS]
    if unknown or not names:
        msg = f"Unknown or empty controllers {unknown}; choose from {', '.join(ALL_CONTROLLERS)}"
        raise ValueError(msg)
    return names


def _write_live_cfg(sumocfg_file: Path, name: str) -> Path:
    """Copy the SUMO config under a per-controller name so each GUI window title shows it."""
    live_cfg = sumocfg_file.parent / f"{LIVE_CFG_PREFIX}{name}.sumocfg"
    live_cfg.write_text(sumocfg_file.read_text(encoding="utf-8"), encoding="utf-8")
    return live_cfg


def _make_policy(
    name: str, settings: TestingSettings, env: Environment, model_path: Path
) -> Policy | None:
    """Build the DQN agent or a baseline controller."""
    if name == DQN:
        return Agent(
            settings=load_model_settings(model_path, settings),
            input_dim=env.state_size,
            output_dim=env.num_actions,
            epsilon=0,
            model_path=model_path,
        )
    return make_controller(ControllerName(name), settings, env)


def live_session(  # noqa: PLR0913
    settings_file: Path,
    controllers: list[str],
    model_path: Path,
    out_path: Path,
    delay: float = 0.0,
    view_radius: float = 120.0,
) -> None:
    """Run several controllers side by side on the same seeded scenario.

    With `gui: true` in the settings, one SUMO-GUI window per controller is tiled on screen,
    zoomed on the junction and stepped in lockstep. Results of every controller and a
    comparison of all of them are saved in `out_path`.

    Args:
        settings_file: Testing settings (scenario, seed, gui).
        controllers: Controller names ("dqn" and/or baselines).
        model_path: Trained model folder (used when "dqn" is in `controllers`).
        out_path: Folder for per-controller results and the comparison.
        delay: GUI delay per simulation step in ms (per window).
        view_radius: Half-size in meters of the area shown around the junction.
    """
    settings = load_testing_settings(settings_file)

    spec = load_network_spec(settings.network, settings.tls_id, settings.signal_program)

    dqn_cells = None
    if DQN in controllers:
        if not (model_path / MODEL_FILE).exists():
            msg = f"Model file not found: '{model_path / MODEL_FILE}' (needed for 'dqn')"
            raise FileNotFoundError(msg)
        dqn_cells = load_model_settings(model_path, settings).cells_per_lane

    # One route file shared by all simulations: identical traffic everywhere.
    generate_routefile(
        seed=settings.episode_seed,
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        turn_chance=settings.turn_chance,
        routes_file=spec.routes_file,
        routes=spec.routes,
    )

    windows = tile_windows(len(controllers), *screen_size())
    runners: list[StepRunner] = []

    try:
        for name, (x, y, w, h) in zip(controllers, windows, strict=True):
            run_path = out_path / name
            run_path.mkdir(parents=True, exist_ok=True)
            env = make_test_environment(
                name,
                settings,
                tripinfo_file=run_path / TRIPINFO_FILE,
                sumocfg_file=_write_live_cfg(spec.sumocfg_file, name),
                **({"cells_per_lane": dqn_cells} if name == DQN and dqn_cells else {}),
            )
            gui_args = (
                [
                    *("--window-pos", f"{x},{y}", "--window-size", f"{w},{h}"),
                    *("--start", "--quit-on-end", "--delay", str(delay)),
                ]
                if settings.gui
                else []
            )
            env.activate(extra_args=gui_args, label=name)
            if settings.gui:
                jx, jy = spec.center
                view = traci.gui.getIDList()[0]
                r = view_radius
                traci.gui.setBoundary(view, jx - r, jy - r, jx + r, jy + r)
            runners.append(StepRunner(name, env, _make_policy(name, settings, env, model_path)))

        logger.info(f"Running {', '.join(controllers)} side by side for {settings.max_steps} steps")
        for _ in range(settings.max_steps):
            for runner in runners:
                traci.switch(runner.label)
                runner.step()

        for runner in runners:
            traci.switch(runner.label)
            runner.finish()
    finally:
        for runner in runners:
            traci.switch(runner.label)
            runner.env.deactivate()

    for runner in runners:
        save_test_results(
            out_path=out_path / runner.label,
            settings_file=settings_file,
            settings=settings,
            controller=runner.label,
            episode_history=runner.history,
            env_stats=runner.env_stats,
        )

    compare_session([out_path / r.label for r in runners], out_path / "comparison")
