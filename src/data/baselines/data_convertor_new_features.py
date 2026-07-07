import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
import os

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

    # Safeguard against folders that don't match the dictionary maps
    if folder.name not in label_map:
        continue
        
    label = label_map[folder.name]

    for file in folder.glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        fragments_lengths = []
        cpg_counts = []

        # Global methylation tokens
        m = 0
        um = 0

        # Region-level aggregates
        region_mean_lengths = []
        region_mean_cpg = []
        region_methyl_ratios = []

        for region in data["regions"].values():
            region_lengths = []
            region_cpgs = []

            # Per-region methylation tokens
            region_m = 0
            region_um = 0

            for fragment in region["fragments"]:
                fl = fragment["fragment_length"]
                cc = fragment["cpg_count"]

                fragments_lengths.append(fl)
                cpg_counts.append(cc)

                region_lengths.append(fl)
                region_cpgs.append(cc)

                # Count internal tokens
                for token in fragment["tokens"]:
                    if token == "<m>":
                        m += 1
                        region_m += 1
                    elif token == "<um>":
                        um += 1
                        region_um += 1

            # Build regional summary arrays
            if len(region_lengths) > 0:
                region_mean_lengths.append(np.mean(region_lengths))
                region_mean_cpg.append(np.mean(region_cpgs))

            region_total = region_m + region_um + 1e-9
            region_methyl_ratios.append(region_m / region_total)

        # ---------------------------------------------------------
        # ADVANCED BIOMARKER ENGINE (SCALE-INVARIANT SIGNALS)
        # ---------------------------------------------------------
        lengths = np.array(fragments_lengths)
        cpgs = np.array(cpg_counts)
        total_methyl = m + um + 1e-9

        # 1. Fragment Size Dynamics & Resolution Metrics
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        skew_len = safe_skew(lengths)
        
        # Volatility index (Coefficient of variation: invariant to batch size)
        len_cv = std_len / (mean_len + 1e-9)

        # Nucleosomal Window Profiles (Captures short tumor-derived shedding shifts)
        ultra_short_ratio = np.mean((lengths >= 90) & (lengths <= 140))
        mononucleosomal_ratio = np.mean((lengths >= 150) & (lengths <= 200))
        nucleosome_ratio_metric = ultra_short_ratio / (mononucleosomal_ratio + 1e-9)

        short_ratio = np.mean(lengths < 150)
        long_ratio = np.mean(lengths > 220)

        # 2. CpG Density & Global Epigenetics
        mean_cpg = np.mean(cpgs)
        std_cpg = np.std(cpgs)
        cpg_density = mean_cpg / (mean_len + 1e-9)

        methyl_ratio = m / total_methyl
        methyl_entropy = -(
            (m / total_methyl) * np.log(m / total_methyl + 1e-9)
            + (um / total_methyl) * np.log(um / total_methyl + 1e-9)
        )

        # 3. Regional Epigenetic Dynamics
        region_mean_lengths = np.array(region_mean_lengths)
        region_mean_cpg = np.array(region_mean_cpg)
        region_methyl_ratios = np.array(region_methyl_ratios)

        region_len_std = np.std(region_mean_lengths) if len(region_mean_lengths) > 0 else 0.0
        region_cpg_std = np.std(region_mean_cpg) if len(region_mean_cpg) > 0 else 0.0
        region_methyl_std = np.std(region_methyl_ratios) if len(region_methyl_ratios) > 0 else 0.0

        # Epigenetic Discordance Delta (Identifies localized hyper/hypo differences)
        if len(region_methyl_ratios) >= 5:
            sorted_ratios = np.sort(region_methyl_ratios)
            top_20_idx = int(len(sorted_ratios) * 0.8)
            bot_20_idx = int(len(sorted_ratios) * 0.2)
            meth_discordance_delta = np.mean(sorted_ratios[top_20_idx:]) - np.mean(sorted_ratios[:bot_20_idx])
        else:
            meth_discordance_delta = 0.0

        # ---------------------------------------------------------
        # ROW ASSEMBLER
        # ---------------------------------------------------------
        rows.append({
            "patient_id": data["patient_id"],
            "label": label,

            # Metadata/Biological Context Ratios
            "enriched_ctdna_fraction": data.get("enriched_ctdna_fraction", 0.0),

            # Size Profiles
            "mean_fragment_length": mean_len,
            "std_fragment_length": std_len,
            "fragment_length_cv": len_cv,
            "skew_fragment_length": skew_len,
            "short_fragment_ratio": short_ratio,
            "long_fragment_ratio": long_ratio,
            "nucleosome_ratio_metric": nucleosome_ratio_metric,

            # CpG Distributions
            "mean_cpg_count": mean_cpg,
            "std_cpg_count": std_cpg,
            "cpg_density": cpg_density,

            # Global Epigenetic Profiles
            "methylation_ratio": methyl_ratio,
            "methylation_entropy": methyl_entropy,
            "meth_discordance_delta": meth_discordance_delta,

            # Structural Dispersions
            "region_length_std": region_len_std,
            "region_cpg_std": region_cpg_std,
            "region_methylation_std": region_methyl_std
        })

        patient_counter += 1

# Generate and store feature matrix
df = pd.DataFrame(rows)

output_path = "src/data/processed/patient_features_improved.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(df.head())
print(f"\nSaved clean bio-feature matrix to {output_path} ✔")
print(f"Total features created: {df.shape[1] - 2} (Excluding ID and Label)")