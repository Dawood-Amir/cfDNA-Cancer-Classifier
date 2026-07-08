import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew

BASE_DIR = "src/data/raw/synthetic_cfdna_output"
OUTPUT_PATH = "src/data/processed/patient_features.csv"

label_map = {
    "label_0_Healthy": 0,
    "label_1_GBM": 1,
    "label_2_LGG": 2,
    "label_3_DMG_H3K27M": 3,
}


def safe_skew(x):
    if len(x) < 3:
        return 0.0
    return float(skew(x))


rows = []

for folder in Path(BASE_DIR).iterdir():
    if not folder.is_dir() or folder.name not in label_map:
        continue

    label = label_map[folder.name]

    for file in folder.glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        fragments_lengths = []
        cpg_counts = []
        m = 0
        um = 0

        region_mean_lengths = []
        region_mean_cpg = []
        region_methyl_ratios = []

        for region in data["regions"].values():
            region_lengths = []
            region_cpgs = []
            region_m = 0
            region_um = 0

            for fragment in region["fragments"]:
                fl = fragment["fragment_length"]
                cc = fragment["cpg_count"]

                fragments_lengths.append(fl)
                cpg_counts.append(cc)
                region_lengths.append(fl)
                region_cpgs.append(cc)

                for token in fragment["tokens"]:
                    if token == "<m>":
                        m += 1
                        region_m += 1
                    elif token == "<um>":
                        um += 1
                        region_um += 1

            if len(region_lengths) > 0:
                region_mean_lengths.append(np.mean(region_lengths))
                region_mean_cpg.append(np.mean(region_cpgs))

            region_total = region_m + region_um + 1e-9
            region_methyl_ratios.append(region_m / region_total)

        lengths = np.array(fragments_lengths)
        cpgs = np.array(cpg_counts)
        total_methyl = m + um + 1e-9

        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))
        skew_len = safe_skew(lengths)
        len_cv = float(std_len / (mean_len + 1e-9))

        ultra_short_ratio = float(np.mean((lengths >= 90) & (lengths <= 140)))
        mononucleosomal_ratio = float(np.mean((lengths >= 150) & (lengths <= 200)))
        nucleosome_ratio_metric = float(
            ultra_short_ratio / (mononucleosomal_ratio + 1e-9)
        )

        short_ratio = float(np.mean(lengths < 150))
        long_ratio = float(np.mean(lengths > 220))

        mean_cpg = float(np.mean(cpgs))
        std_cpg = float(np.std(cpgs))
        cpg_density = float(mean_cpg / (mean_len + 1e-9))

        methyl_ratio = float(m / total_methyl)
        methyl_entropy = -float(
            (m / total_methyl) * np.log(m / total_methyl + 1e-9)
            + (um / total_methyl) * np.log(um / total_methyl + 1e-9)
        )

        region_mean_lengths = np.array(region_mean_lengths)
        region_mean_cpg = np.array(region_mean_cpg)
        region_methyl_ratios = np.array(region_methyl_ratios)

        region_len_std = (
            float(np.std(region_mean_lengths))
            if len(region_mean_lengths) > 0
            else 0.0
        )
        region_cpg_std = (
            float(np.std(region_mean_cpg)) if len(region_mean_cpg) > 0 else 0.0
        )
        region_methyl_std = (
            float(np.std(region_methyl_ratios))
            if len(region_methyl_ratios) > 0
            else 0.0
        )

        if len(region_methyl_ratios) >= 5:
            sorted_ratios = np.sort(region_methyl_ratios)
            top_20_idx = int(len(sorted_ratios) * 0.8)
            bot_20_idx = int(len(sorted_ratios) * 0.2)
            meth_discordance_delta = float(
                np.mean(sorted_ratios[top_20_idx:])
                - np.mean(sorted_ratios[:bot_20_idx])
            )
        else:
            meth_discordance_delta = 0.0

        rows.append(
            {
                "patient_id": data["patient_id"],
                "label": label,
                "mean_fragment_length": mean_len,
                "std_fragment_length": std_len,
                "fragment_length_cv": len_cv,
                "skew_fragment_length": skew_len,
                "short_fragment_ratio": short_ratio,
                "long_fragment_ratio": long_ratio,
                "nucleosome_ratio_metric": nucleosome_ratio_metric,
                "mean_cpg_count": mean_cpg,
                "std_cpg_count": std_cpg,
                "cpg_density": cpg_density,
                "methylation_ratio": methyl_ratio,
                "methylation_entropy": methyl_entropy,
                "meth_discordance_delta": meth_discordance_delta,
                "region_length_std": region_len_std,
                "region_cpg_std": region_cpg_std,
                "region_methylation_std": region_methyl_std,
            }
        )

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(df.head())
print(f"\nSaved clean bio-feature matrix to {OUTPUT_PATH}")
print(f"Total features created: {df.shape[1] - 2} (Excluding ID and Label)")