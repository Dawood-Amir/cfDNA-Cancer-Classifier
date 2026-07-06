from experiments.run_experiment import run_experiment
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"

run_experiment(
    CONFIG_PATH,
    seed=10
)

# import pandas as pd

# # Load the feature set you just generated
# df = pd.read_csv("src/data/processed/patient_features_improved2.csv")

# # Print the mean value of key features for each class
# features_to_check = [
#     "mean_fragment_length", 
#     "short_fragment_ratio", 
#     "methylation_ratio", 
#     "motif_CCCA"
# ]

# print("========== DATASET DIAGNOSTIC CRUISE ==========")
# for feat in features_to_check:
#     if feat in df.columns:
#         print(f"\nAverage {feat} per class:")
#         print(df.groupby("label")[feat].mean())