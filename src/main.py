from experiments.run_experiment import run_experiment
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"

run_experiment(
    CONFIG_PATH,
    seed=42
)