from pathlib import Path

import pytest
from pydantic import ValidationError

from tlcs.settings import TestingSettings as Settings
from tlcs.settings import load_testing_settings, load_training_settings

SETTINGS_DIR = Path(__file__).parents[1] / "settings"

BASE_TESTING = {
    "gui": False,
    "max_steps": 100,
    "n_cars_generated": 10,
    "episode_seed": 1,
    "yellow_duration": 4,
    "green_duration": 10,
    "turn_chance": 0.25,
    "gamma": 0.75,
}


def test_shipped_settings_still_load() -> None:
    training = load_training_settings(SETTINGS_DIR / "training_settings.yaml")
    assert (training.network, training.cells_per_lane) == ("original", 5)
    for name in ("t_junction_training.yaml", "t_junction_testing.yaml"):
        loader = load_training_settings if "training" in name else load_testing_settings
        assert loader(SETTINGS_DIR / name).network == "t_junction"
    settings = load_testing_settings(SETTINGS_DIR / "testing_settings.yaml")
    # Baseline options are optional and default when absent.
    assert settings.fixed_green_durations is None
    assert settings.actuated_max_green == 60


def test_fixed_green_durations_accepts_any_length() -> None:
    # The number of phases depends on the network; it is checked when the controller is built.
    for durations in ([30, 30], [30, 10, 30, 10]):
        settings = Settings.model_validate({**BASE_TESTING, "fixed_green_durations": durations})
        assert settings.fixed_green_durations == durations


def test_fixed_green_durations_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({**BASE_TESTING, "fixed_green_durations": [30, 0]})


def test_network_settings_default_to_original() -> None:
    settings = Settings.model_validate(BASE_TESTING)
    assert (settings.network, settings.tls_id) == ("original", None)


def test_old_settings_with_sumocfg_file_still_load() -> None:
    # Settings saved before the `network` option had `sumocfg_file`; it is now ignored.
    old = {**BASE_TESTING, "sumocfg_file": "intersection/sumo_config.sumocfg"}
    assert Settings.model_validate(old).network == "original"


def test_actuated_max_green_must_cover_min_green() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({**BASE_TESTING, "actuated_max_green": 5})


def test_koteshwor_demand_presets_differ_only_in_demand() -> None:
    base = load_testing_settings(SETTINGS_DIR / "koteshwor_testing.yaml").model_dump()
    demands = {}
    for name in ("low", "medium", "high"):
        preset = load_testing_settings(SETTINGS_DIR / f"koteshwor_testing_{name}.yaml").model_dump()
        demands[name] = preset.pop("n_cars_generated")
        assert not preset.pop("gui")
        assert preset == {k: v for k, v in base.items() if k not in {"gui", "n_cars_generated"}}
    assert demands["low"] < demands["medium"] == base["n_cars_generated"] < demands["high"]


def test_checkpoint_every_defaults_to_100() -> None:
    training = load_training_settings(SETTINGS_DIR / "koteshwor_training.yaml")
    assert training.checkpoint_every == 100
