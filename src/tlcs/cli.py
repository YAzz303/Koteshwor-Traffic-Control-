from pathlib import Path
from typing import Annotated

import typer

from tlcs.baselines import ControllerName, baseline_session
from tlcs.compare import compare_session
from tlcs.constants import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_COMPARE_PATH,
    DEFAULT_LIVE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SETTINGS_PATH,
    DEFAULT_TEST_FOLDER,
    TESTING_SETTINGS_FILE,
    TRAINING_SETTINGS_FILE,
)
from tlcs.live import ALL_CONTROLLERS, live_session, parse_controllers
from tlcs.main import testing_session, training_session

app = typer.Typer(
    help="Train and run TLCS.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _confirm_overwrite_directory(
    directory: Path,
    overwrite_message: str,
    cancel_message: str,
) -> None:
    """Ask the user to confirm overwriting an existing directory.

    Args:
        directory: The directory that potentially will be overwritten.
        overwrite_message: The message prompting the user to confirm overwrite.
        cancel_message: The message displayed if the user cancels.
    """
    typer.echo(f"⚠️  The folder '{directory}' already exists.")
    confirm = typer.confirm(overwrite_message, default=False)
    if not confirm:
        typer.echo(cancel_message)
        raise typer.Abort


def check_training_path(out_path: Path) -> None:
    """Ensure training output path is safe to use.

    Args:
        out_path: Directory where training outputs and model will be saved.
    """
    if out_path.exists():
        _confirm_overwrite_directory(
            directory=out_path,
            overwrite_message=(
                "Continuing will possibly overwrite the existing training data and model. "
                "Do you want to continue?"
            ),
            cancel_message="Training cancelled.",
        )


def check_testing_path(model_path: Path, test_name: str) -> None:
    """Ensure testing model path and test folder are valid.

    Args:
        model_path: Path to the directory containing the trained model.
        test_name: Name of the test folder to create within the model directory.
    """
    model_files = list(model_path.glob("*.pt"))

    if not model_files:
        typer.echo(f"Model file (*.pt) not found in model path: '{model_path}'.")
        raise typer.Abort

    test_folder = model_path / test_name

    if test_folder.exists():
        _confirm_overwrite_directory(
            directory=test_folder,
            overwrite_message=(
                "Continuing will overwrite the content of the test folder. Do you want to continue?"
            ),
            cancel_message="Testing cancelled.",
        )


@app.command(name="train", help="Train a new TLCS model using the specified settings file.")
def cmd_train(
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing training parameters.",
        ),
    ] = DEFAULT_SETTINGS_PATH / TRAINING_SETTINGS_FILE,
    out_path: Annotated[
        Path,
        typer.Option(
            help="Directory where training outputs and trained model will be saved.",
        ),
    ] = DEFAULT_MODEL_PATH,
) -> None:
    """CLI command to train a TLCS model.

    Args:
        settings_file: Path to the YAML file with training parameters.
        out_path: Output directory for training artifacts and the trained model.
    """
    check_training_path(out_path)
    training_session(settings_file=settings_file, out_path=out_path)


@app.command(name="test", help="Run a simulation test using a trained TLCS model.")
def cmd_test(
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing testing parameters.",
        ),
    ] = DEFAULT_SETTINGS_PATH / TESTING_SETTINGS_FILE,
    model_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the directory containing the trained TLCS model to test.",
        ),
    ] = DEFAULT_MODEL_PATH,
    test_name: Annotated[
        str,
        typer.Option(
            help="The name of the test folder.",
        ),
    ] = DEFAULT_TEST_FOLDER,
) -> None:
    """CLI command to run a simulation test using a trained TLCS model.

    Args:
        settings_file: Path to the YAML file with testing parameters.
        model_path: Path to the directory containing the trained model.
        test_name: Name of the test folder created under the model directory.
    """
    check_testing_path(model_path=model_path, test_name=test_name)
    testing_session(settings_file=settings_file, model_path=model_path, test_name=test_name)


@app.command(
    name="baseline",
    help="Run a non-learning baseline controller on the same seeded scenario as `tlcs test`.",
)
def cmd_baseline(
    controller: Annotated[
        ControllerName,
        typer.Option(help="Baseline controller to run.", case_sensitive=False),
    ],
    test_name: Annotated[
        str,
        typer.Option(help="Name of the results folder (created inside --out-path)."),
    ],
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing testing parameters.",
        ),
    ] = DEFAULT_SETTINGS_PATH / TESTING_SETTINGS_FILE,
    out_path: Annotated[
        Path,
        typer.Option(help="Parent directory for baseline results."),
    ] = DEFAULT_BASELINE_PATH,
) -> None:
    """CLI command to run a baseline controller.

    Args:
        controller: Which baseline controller to run.
        test_name: Name of the results folder.
        settings_file: Path to the YAML file with testing parameters.
        out_path: Parent directory for baseline results.
    """
    run_path = out_path / test_name
    if run_path.exists():
        _confirm_overwrite_directory(
            directory=run_path,
            overwrite_message="Continuing will overwrite its content. Do you want to continue?",
            cancel_message="Baseline run cancelled.",
        )
    baseline_session(controller=controller, settings_file=settings_file, out_path=run_path)


@app.command(
    name="compare",
    help=(
        "Compare saved test runs: one plot per metric plus a summary table. "
        "Usage: tlcs compare --runs RUN_DIR [RUN_DIR ...]"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": False},
)
def cmd_compare(
    ctx: typer.Context,
    runs: Annotated[
        list[Path],
        typer.Option(
            exists=True,
            file_okay=False,
            help="Test run folders (from `tlcs test` or `tlcs baseline`). Space-separated.",
        ),
    ],
    out_path: Annotated[
        Path,
        typer.Option(help="Directory where comparison plots and summary.csv are saved."),
    ] = DEFAULT_COMPARE_PATH,
    smooth: Annotated[
        int,
        typer.Option(min=1, help="Moving-average window (steps) for the queue plot; 1 = raw."),
    ] = 60,
    labels: Annotated[
        str | None,
        typer.Option(
            help=(
                "Comma-separated label per run, in --runs order (default: folder names). "
                "With runs at several demands, give one controller the same label at each."
            ),
        ),
    ] = None,
) -> None:
    """CLI command to compare several test runs.

    Args:
        ctx: Typer context; extra positional values after --runs are treated as more runs.
        runs: Test run folders.
        out_path: Output directory for comparison artifacts.
        smooth: Moving-average window for the queue plot.
        labels: Comma-separated display labels, one per run.
    """
    run_paths = list(runs)
    for extra in ctx.args:
        path = Path(extra)
        if not path.is_dir():
            typer.echo(f"Run folder not found: '{extra}'.")
            raise typer.Abort
        run_paths.append(path)

    label_list = [label.strip() for label in labels.split(",")] if labels else None
    try:
        compare_session(run_paths=run_paths, out_path=out_path, smooth=smooth, labels=label_list)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Abort from e


@app.command(
    name="live",
    help=(
        "Watch several controllers side by side: one SUMO-GUI window per method, all on the "
        "same seeded traffic and stepped together. Also saves results and a comparison."
    ),
)
def cmd_live(  # noqa: PLR0913
    controllers: Annotated[
        str,
        typer.Option(help=f"Comma-separated controllers from: {', '.join(ALL_CONTROLLERS)}."),
    ] = ",".join(ALL_CONTROLLERS),
    model_path: Annotated[
        Path,
        typer.Option(help="Trained model folder (needed when 'dqn' is included)."),
    ] = DEFAULT_MODEL_PATH,
    settings_file: Annotated[
        Path,
        typer.Option(exists=True, help="Testing settings (scenario, seed and gui on/off)."),
    ] = DEFAULT_SETTINGS_PATH / TESTING_SETTINGS_FILE,
    out_path: Annotated[
        Path,
        typer.Option(help="Folder for per-controller results and the comparison."),
    ] = DEFAULT_LIVE_PATH,
    delay: Annotated[
        float,
        typer.Option(min=0, help="GUI delay per simulation step in ms (per window)."),
    ] = 0.0,
    view_radius: Annotated[
        float,
        typer.Option(min=10, help="Half-size (m) of the area shown around each junction."),
    ] = 120.0,
) -> None:
    """CLI command to run controllers side by side.

    Args:
        controllers: Comma-separated controller names.
        model_path: Trained model folder.
        settings_file: Testing settings file.
        out_path: Output folder.
        delay: GUI delay per step in ms.
        view_radius: Visible area around each junction in meters.
    """
    try:
        names = parse_controllers(controllers)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Abort from e

    if out_path.exists():
        _confirm_overwrite_directory(
            directory=out_path,
            overwrite_message="Continuing will overwrite its content. Do you want to continue?",
            cancel_message="Live run cancelled.",
        )
    live_session(
        settings_file=settings_file,
        controllers=names,
        model_path=model_path,
        out_path=out_path,
        delay=delay,
        view_radius=view_radius,
    )


if __name__ == "__main__":
    app()
