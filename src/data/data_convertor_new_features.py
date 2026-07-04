import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew

BASE_DIR = "src/data/raw/synthetic_cfdna_output"

label_map = {
    "label_0_Healthy": 0,
    "label_1_GBM": 1,
    "label_2_LGG": 2,
    "label_3_DMG_H3K27M": 3
}

rows = []
patient_counter = 0


def safe_skew(x):
    if len(x) < 3:
        return 0.0
    return skew(x)


for folder in Path(BASE_DIR).iterdir():
    if not folder.is_dir():
        continue

    label = label_map[folder.name]

    for file in folder.glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        fragments_lengths = []
        cpg_counts = []

        # global methylation
        m = 0
        um = 0

        # region-level stats
        region_mean_lengths = []
        region_mean_cpg = []

        # 🔥 NEW: region methylation stats
        region_m_counts = []
        region_um_counts = []
        region_methyl_ratios = []

        for region in data["regions"].values():

            region_lengths = []
            region_cpgs = []

            # 🔥 NEW per-region methylation
            region_m = 0
            region_um = 0

            for fragment in region["fragments"]:
                fl = fragment["fragment_length"]
                cc = fragment["cpg_count"]

                fragments_lengths.append(fl)
                cpg_counts.append(cc)

                region_lengths.append(fl)
                region_cpgs.append(cc)

                # methylation tokens
                for token in fragment["tokens"]:
                    if token == "<m>":
                        m += 1
                        region_m += 1
                    elif token == "<um>":
                        um += 1
                        region_um += 1

            # region summaries
            if len(region_lengths) > 0:
                region_mean_lengths.append(np.mean(region_lengths))
                region_mean_cpg.append(np.mean(region_cpgs))

            # 🔥 NEW: store region methylation
            region_total = region_m + region_um + 1e-9

            region_m_counts.append(region_m)
            region_um_counts.append(region_um)
            region_methyl_ratios.append(region_m / region_total)

        # -----------------------
        # GLOBAL FEATURES
        # -----------------------

        lengths = np.array(fragments_lengths)
        cpgs = np.array(cpg_counts)

        total_methyl = m + um + 1e-9

        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        skew_len = safe_skew(lengths)

        short_ratio = np.mean(lengths < 150)
        long_ratio = np.mean(lengths > 220)

        mean_cpg = np.mean(cpgs)
        std_cpg = np.std(cpgs)
        cpg_density = mean_cpg / (mean_len + 1e-9)

        methyl_ratio = m / total_methyl

        methyl_entropy = -(
            (m / total_methyl) * np.log(m / total_methyl + 1e-9)
            + (um / total_methyl) * np.log(um / total_methyl + 1e-9)
        )

        region_mean_lengths = np.array(region_mean_lengths)
        region_mean_cpg = np.array(region_mean_cpg)

        region_len_std = np.std(region_mean_lengths) if len(region_mean_lengths) > 0 else 0.0
        region_cpg_std = np.std(region_mean_cpg) if len(region_mean_cpg) > 0 else 0.0

        region_m_counts = np.array(region_m_counts)
        region_um_counts = np.array(region_um_counts)
        region_methyl_ratios = np.array(region_methyl_ratios)

        region_m_mean = np.mean(region_m_counts) if len(region_m_counts) > 0 else 0.0
        region_um_mean = np.mean(region_um_counts) if len(region_um_counts) > 0 else 0.0
        region_methyl_std = np.std(region_methyl_ratios) if len(region_methyl_ratios) > 0 else 0.0

        # -----------------------
        # FINAL ROW
        # -----------------------

        rows.append({
            "patient_id": data["patient_id"],
            "label": label,

            # structure
            "n_regions": len(data["regions"]),
            "n_fragments": data["n_fragments"],

            # fragment distribution
            "mean_fragment_length": mean_len,
            "std_fragment_length": std_len,
            "skew_fragment_length": skew_len,
            "short_fragment_ratio": short_ratio,
            "long_fragment_ratio": long_ratio,

            # CpG features
            "mean_cpg_count": mean_cpg,
            "std_cpg_count": std_cpg,
            "cpg_density": cpg_density,

            # methylation (global)
            "methylation_ratio": methyl_ratio,
            "methylation_entropy": methyl_entropy,

            # region structure
            "region_length_std": region_len_std,
            "region_cpg_std": region_cpg_std,

            # region methylation features
            "region_m_mean": region_m_mean,
            "region_um_mean": region_um_mean,
            "region_methylation_std": region_methyl_std
        })

        patient_counter += 1


df = pd.DataFrame(rows)

df.to_csv(
    "src/data/processed/patient_features_improved.csv",
    index=False
)

print(df.head())
print("\nSaved improved feature set ✔")


