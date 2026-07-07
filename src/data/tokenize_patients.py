import os
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from tokenizer import CFDNATokenizer

RAW_DIR = "src/data/raw/synthetic_cfdna_output"
SAVE_DIR = "src/data/processed/patient_tensors"

os.makedirs(SAVE_DIR, exist_ok=True)

LABEL_MAP = {
    "label_0_Healthy":0,
    "label_1_GBM":1,
    "label_2_LGG":2,
    "label_3_DMG_H3K27M":3
}

tokenizer = CFDNATokenizer()

manifest=[]

for class_folder in Path(RAW_DIR).iterdir():

    if not class_folder.is_dir():
        continue

    label=LABEL_MAP[class_folder.name]

    print(f"\nProcessing {class_folder.name}")

    for patient_file in tqdm(class_folder.glob("*.json")):

        with open(patient_file) as f:
            patient=json.load(f)

        patient_regions=[]

        ####################################################
        # Preserve EVERY region
        ####################################################

        for region_name in sorted(patient["regions"].keys()):

            region=patient["regions"][region_name]

            merged_tokens=[]

            ################################################
            # concatenate fragments
            ################################################

            for fragment in region["fragments"]:

                merged_tokens.extend(
                    fragment["tokens"]
                )

            encoded=tokenizer.encode(
                merged_tokens
            )

            patient_regions.append({

                "tokens":encoded,

                "chromosome":region["chromosome"],

                "genomic_start":region["genomic_start"],

                "num_fragments":len(region["fragments"])

            })

        save_name=patient_file.stem+".pt"
        total_tokens = sum(
            len(region["tokens"])
            for region in patient_regions
        )
        torch.save(

            {

                "patient_id":patient["patient_id"],
                "label":label,
                "total_tokens": total_tokens,
                "regions":patient_regions

            },

            os.path.join(
                SAVE_DIR,
                save_name
            )

        )

        manifest.append({

            "filename":save_name,

            "label":label

        })

pd.DataFrame(manifest).to_csv(

    os.path.join(
        SAVE_DIR,
        "manifest.csv"
    ),

    index=False

)

print("\nFinished preprocessing.")