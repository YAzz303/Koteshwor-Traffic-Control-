import os
from pathlib import Path

# Render plots off-screen during tests.
os.environ.setdefault("MPLBACKEND", "Agg")

# Network folders are resolved relative to the project root (intersection/<network>).
os.chdir(Path(__file__).resolve().parents[1])
