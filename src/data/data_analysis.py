import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "src/data/processed/region_tensors"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

df = pd.read_csv(MANIFEST_PATH)

print("\n==============================")
print("TOTAL PATIENTS:", len(df))
print("==============================\n")

print("CLASS DISTRIBUTION:")
print(df["label"].value_counts())
print("\n")

# ==============================
# STORAGE
# ==============================
stats = {
    "label": [],
    "num_regions": [],
    "mean_fragments": [],
    "mean_length": [],
    "mean_cpg": [],
    "mean_methyl_ratio": [],
    "mean_entropy": []
}

# ==============================
# LOAD DATA
# ==============================
for _, row in df.iterrows():
    arr = np.load(os.path.join(DATA_DIR, row["filename"]))
    label = row["label"]

    stats["label"].append(label)
    stats["num_regions"].append(arr.shape[0])

    stats["mean_fragments"].append(np.mean(arr[:, 0]))
    stats["mean_length"].append(np.mean(arr[:, 1]))
    stats["mean_cpg"].append(np.mean(arr[:, 4]))
    stats["mean_methyl_ratio"].append(np.mean(arr[:, 8]))
    stats["mean_entropy"].append(np.mean(arr[:, 9]))

stats_df = pd.DataFrame(stats)

# ==============================
# PRINT GROUP SUMMARY
# ==============================
print("\n==============================")
print("MEAN VALUES PER CLASS")
print("==============================\n")
print(stats_df.groupby("label").mean())

# ==============================
# SIMPLE INTERPRETATION HELP
# ==============================
print("\n==============================")
print("QUICK INSIGHT CHECK")
print("==============================\n")

for col in stats_df.columns[1:]:
    print(f"\nFeature: {col}")
    print(stats_df.groupby("label")[col].mean())

# ==============================
# PLOTS
# ==============================
features = stats_df.columns[1:]

for feat in features:
    plt.figure()
    plt.title(feat)

    for lbl in sorted(stats_df["label"].unique()):
        plt.hist(
            stats_df[stats_df["label"] == lbl][feat],
            bins=30,
            alpha=0.5,
            label=f"Class {lbl}"
        )

    plt.legend()
    plt.show()

    print(f"\nDisplayed plot: {feat}")