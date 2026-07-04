"""
=====================================================================
RegionDataset
=====================================================================

Instead of converting every patient into ONE row of a CSV,
this dataset reads the original JSON files directly.

Each patient becomes

(number_of_regions, number_of_features_per_region)

Example:

Patient A
torch.Size([248, 15])

Patient B
torch.Size([263, 15])

Patient C
torch.Size([271, 15])

Notice that the number of regions is different for every patient.
We will solve that later using a custom collate function.

=====================================================================
"""



import json 
import numpy as np
import torch 
from pathlib import Path
from scipy.stats import skew
from torch.utils.data import Dataset

LABEL_MAP = {
    "label_0_Healthy": 0,
    "label_1_GBM": 1,
    "label_2_LGG": 2,
    "label_3_DMG_H3K27M": 3

}

class RegionDataset(Dataset):
    """
    PyTorch Dataset.

    Every sample returned is ONE patient.

    X = tensor(num_regions, num_features)

    y = class label
    """

    def __init__(self, base_dir: str):
        self.samples = []
        base_dir = Path(base_dir)

        for folder in base_dir.iterdir():
            if not folder.is_dir():
                continue

            label = LABEL_MAP[folder.name]

            for json_file in folder.glob("*.json"):
                self.samples.append((json_file, label))


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Returns ONE patient.

        Output:

        region_tensor

        shape:

        (number_of_regions,
         number_of_region_features)

        label
        """
        json_file,label =self.samples[index]

        with open(json_file) as f:
            patient = json.load(f)
        
        # Every region will become ONE feature vector
        region_vectors = []

        # Loop over every genomic region
        for region in patient["regions"].values():
            fragments = region["fragments"] # got fragment here in a region 
            fragment_lengths =[]

            cpg_counts =[]
            methylated=0
            unmethylated=0

            fragment_ratios=[]

            # Loop over fragments INSIDE this region

            for fragment in fragments:
                fragment_lengths.append(fragment["fragment_length"])

                cpg_counts.append(fragment["cpg_count"])
                m =0
                um=0

                for token in fragment["tokens"]:
                    if token == "<m>":
                        methylated += 1
                        m+=1
                    elif token == "<um>":
                        unmethylated += 1
                        um+=1

                total = m + um 

                if total > 0:
                    fragment_ratios.append(m / total)


            #Region-level statistics
            lengths = np.asarray(fragment_lengths)
            cpg = np.asarray(cpg_counts)    

            total_tokens = methylated + unmethylated

            if total_tokens > 0:
                methyl_ratio  = methylated / total_tokens

                entropy = ( -methyl_ratio * np.log(methyl_ratio+ 1e-9)
                            - (1-methyl_ratio) * np.log(1-methyl_ratio + 1e-9))
                
            else:
                methyl_ratio = 0.0
                entropy = 0.0

            # ONE REGION
            # represented by ONE vector
            
            region_vector = [
                len(fragment_lengths), # number of fragments in this region
                np.mean(lengths) if len(lengths) > 0 else 0.0,
                np.std(lengths) if len(lengths) > 0 else 0.0,
                skew(lengths) if len(lengths) > 2 and np.std(lengths) > 0 else 0.0,

                np.mean(cpg) if len(cpg) > 0 else 0.0,
                np.std(cpg) if len(cpg) > 0  else 0.0,
                skew(cpg) if len(cpg) > 2 and np.std(cpg) > 0 else 0.0,

                np.mean(cpg / (lengths + 1e-9)) if len(lengths) > 0 else 0.0,

                methyl_ratio,
                entropy,

                np.mean(fragment_ratios) if len(fragment_ratios) > 0 else 0.0,

                np.std(fragment_ratios) if len(fragment_ratios) > 0 else 0.0,

                np.mean(lengths < 150),
                np.mean(lengths > 220),
                np.min(lengths) if len(lengths) > 0 else 0.0,
                np.max(lengths) if len(lengths) > 0 else 0.0

                ]
            
            region_vectors.append(region_vector)

        
        #Convert to PyTorch tensor
        region_tensor = torch.tensor(region_vectors ,dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return region_tensor, label_tensor
    


if __name__ == "__main__":

    dataset = RegionDataset("src/data/raw/synthetic_cfdna_output")

    X, y = dataset[900]

    print("Patient tensor shape:", X.shape)
    print("Label:", y)

    print()

    print("First region:")

    print(X[0])