import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import skew
import os

BASE_DIR = "src/data/raw/synthetic_cfdna_output"
OUTPUT_DIR = "src/data/processed/region_tensors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP = {"label_0_Healthy": 0, "label_1_GBM": 1, "label_2_LGG": 2, "label_3_DMG_H3K27M": 3}

print("Starting high-speed region pre-extraction. This runs ONCE...")

patient_list = []

for folder in Path(BASE_DIR).iterdir():
    if not folder.is_dir(): continue
    label = LABEL_MAP[folder.name]

    for json_file in folder.glob("*.json"):
        with open(json_file) as f:
            patient = json.load(f)
        
        region_vectors = []
        for region in patient["regions"].values():
            fragments = region.get("fragments", [])
            if len(fragments) == 0:
                # Direct safe default for missing regions to prevent NaN crashes
                region_vectors.append([0.0] * 16)
                continue

            lengths = np.array([f["fragment_length"] for f in fragments], dtype=np.float32)
            cpg = np.array([f["cpg_count"] for f in fragments], dtype=np.float32)
            
            # Methylation flags profiling
            methylated = 0
            unmethylated = 0
            fragment_ratios = []
            
            for fragment in fragments:
                m = sum(1 for t in fragment["tokens"] if t == "<m>")
                um = sum(1 for t in fragment["tokens"] if t == "<um>")
                methylated += m
                unmethylated += um
                if (m + um) > 0:
                    fragment_ratios.append(m / (m + um))

            total_tokens = methylated + unmethylated
            if total_tokens > 0:
                methyl_ratio = methylated / total_tokens
                entropy = (-methyl_ratio * np.log(methyl_ratio + 1e-9) 
                           - (1 - methyl_ratio) * np.log(1 - methyl_ratio + 1e-9))
            else:
                methyl_ratio, entropy = 0.0, 0.0

            std_len = np.std(lengths)
            std_cpg = np.std(cpg)

            # Build single highly optimized vector row
            region_vector = [
                float(len(lengths)),
                float(np.mean(lengths)),
                float(std_len),
                float(skew(lengths)) if len(lengths) > 2 and std_len > 0 else 0.0,
                float(np.mean(cpg)),
                float(std_cpg),
                float(skew(cpg)) if len(cpg) > 2 and std_cpg > 0 else 0.0,
                float(np.mean(cpg / (lengths + 1e-9))),
                float(methyl_ratio),
                float(entropy),
                float(np.mean(fragment_ratios)) if fragment_ratios else 0.0,
                float(np.std(fragment_ratios)) if fragment_ratios else 0.0,
                float(np.mean(lengths < 150)),
                float(np.mean(lengths > 220)),
                float(np.min(lengths)),
                float(np.max(lengths))
            ]
            region_vectors.append(region_vector)
        
        # Save each array individually under binary naming conventions
        np_arr = np.array(region_vectors, dtype=np.float32)
        out_filename = f"{json_file.stem}.npy"
        np.save(os.path.join(OUTPUT_DIR, out_filename), np_arr)
        
        patient_list.append({"filename": out_filename, "label": label})

# Save index meta reference file
import pandas as pd
pd.DataFrame(patient_list).to_csv(os.path.join(OUTPUT_DIR, "manifest.csv"), index=False)
print(f"Extraction complete! Saved processed matrices to {OUTPUT_DIR}")