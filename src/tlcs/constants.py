from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and filenames
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS_PATH = Path("settings")
DEFAULT_MODEL_PATH = Path("model")

TRAINING_SETTINGS_FILE = Path("training_settings.yaml")
TESTING_SETTINGS_FILE = Path("testing_settings.yaml")

MODEL_FILE = Path("trained_model.pt")

DEFAULT_TEST_FOLDER = "test"

# ---------------------------------------------------------------------------
# SUMO route file configuration
# ---------------------------------------------------------------------------

# Program ID of the generated actuated traffic-light program (actuated baseline).
ACTUATED_PROGRAM_ID = "actuated"

# Default output folders for baseline runs and comparisons.
DEFAULT_BASELINE_PATH = Path("baselines")
DEFAULT_COMPARE_PATH = Path("comparison")
DEFAULT_LIVE_PATH = Path("live")

# Extra files written by every test run (DQN or baseline), used by `tlcs compare`.
TRIPINFO_FILE = Path("tripinfo.xml")
REWARD_STEPS_FILE = Path("reward_steps_data.txt")
RUN_INFO_FILE = Path("run_info.json")
