import json 
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = "data/raw/synthetic_cfdna_output"

label_map={
    "label_0_Healthy": 0,
    "label_1_GBM": 1,
    "label_2_LGG": 2,
    "label_3_DMG_H3K27M": 3
}

rows=[]

patient_counter = 0

for folder in Path(BASE_DIR).iterdir():
    if not folder.is_dir():
        continue

    label = label_map[folder.name]

    for file in folder.glob("*.json"):
        with open(file) as f:
            data = json.load(f) #Got data here

        fragments_lentghs =[]
        cpg_counts =[]
        m=0
        um=0

        for region in data['regions'].values(): # in regions dict
            
            for fragment in region["fragments"]:
                fragments_lentghs.append(fragment["fragment_length"])
                cpg_counts.append(fragment["cpg_count"])

                for token in fragment["tokens"]:
                    if token == "<m>":
                        m+=1
                    elif token == "<um>":
                        um+=1

        #"patient_id": f"{label}_{patient_counter:04d}"
        # "raw_ctdna_fraction": data["raw_ctdna_fraction"],
        #"enriched_ctdna_fraction":data["enriched_ctdna_fraction"], 
        rows.append({
            "patient_id": data["patient_id"],
            "label": label,
            

            "n_regions": len(data["regions"]),
            "n_fragments": data["n_fragments"],

            "mean_fragment_length": np.mean(fragments_lentghs),
            "std_fragment_length": np.std(fragments_lentghs),

            "mean_cpg_count": np.mean(cpg_counts),

            "methylation_ratio": m / (m + um + 1e-9)
        })
        patient_counter+=1

df = pd.DataFrame(rows)

df.to_csv("data/processed/patient_features_without_ctdna.csv", index=False)
print(df.head())
