
from pathlib import Path
import argparse
import os
from pathlib import Path
import yaml
from experiments.run_experiment import run_experiment

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def main():
    # 1. Setup CLI Argument Parser
    parser = argparse.ArgumentParser(
        description="Run cfDNA Cancer Classifier Baseline Experiments."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ffn", "cnn", "xgboost" ,"resffn"],
        help="Specify the baseline architecture to train and evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed initialization for reproducibility.",
    )

    args = parser.parse_args()

    # 2. Open and dynamically update the YAML configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Override the model type with the command line argument
    config["model"]["type"] = args.model

    # Save the temporary modified configuration so run_experiment reads the target choice
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

    print(f" Initializing pipeline run...")
    print(f" Selected Architecture: {args.model.upper()}")
    print(f" Execution Seed      : {args.seed}")

    # 3. Launch the standardized experiment loop
    run_experiment(str(CONFIG_PATH), seed=args.seed)


if __name__ == "__main__":
    main()



# CONFIG_PATH = Path(__file__).parent / "config.yaml"

# run_experiment(
#     CONFIG_PATH,
#     seed=10
# )
