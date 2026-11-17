from pathlib import Path

from tlcs.constants import MODEL_FILE, TRAINING_SETTINGS_FILE
from tlcs.main import checkpoint_dir, is_checkpoint_episode, save_checkpoint


def test_is_checkpoint_episode() -> None:
    assert [n for n in range(1, 301) if is_checkpoint_episode(n, 100)] == [100, 200, 300]
    assert not any(is_checkpoint_episode(n, 0) for n in range(1, 301))


def test_checkpoint_dir_is_sortable() -> None:
    assert checkpoint_dir(Path("run"), 100) == Path("run/checkpoints/episode_0100")


class _FakeAgent:
    def save_model(self, out_path: Path) -> None:
        (out_path / MODEL_FILE).write_bytes(b"model")


def test_save_checkpoint_writes_a_testable_folder(tmp_path: Path) -> None:
    settings_file = tmp_path / "train.yaml"
    settings_file.write_text("total_episodes: 1000\n", encoding="utf-8")
    path = save_checkpoint(_FakeAgent(), settings_file, tmp_path / "run", 200)  # type: ignore[arg-type]
    assert path == tmp_path / "run" / "checkpoints" / "episode_0200"
    assert (path / MODEL_FILE).read_bytes() == b"model"
    assert (path / TRAINING_SETTINGS_FILE).read_text(encoding="utf-8") == "total_episodes: 1000\n"
