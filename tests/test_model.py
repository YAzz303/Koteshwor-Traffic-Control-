from pathlib import Path

import numpy as np
import pytest

from tlcs.model import Model


def _model(input_dim: int, output_dim: int, model_path: Path | None = None) -> Model:
    return Model(
        num_layers=1,
        width=8,
        learning_rate=0.001,
        input_dim=input_dim,
        output_dim=output_dim,
        model_path=model_path,
    )


def test_network_sizes_come_from_the_environment() -> None:
    model = _model(input_dim=30, output_dim=2)
    assert model.predict_one(np.zeros(30)).shape == (1, 2)


def test_loaded_model_must_match_environment_sizes(tmp_path: Path) -> None:
    _model(input_dim=30, output_dim=2).save_model(tmp_path)
    assert _model(30, 2, model_path=tmp_path).predict_one(np.zeros(30)).shape == (1, 2)
    with pytest.raises(ValueError, match="expects 30 inputs and 2 actions"):
        _model(80, 4, model_path=tmp_path)
