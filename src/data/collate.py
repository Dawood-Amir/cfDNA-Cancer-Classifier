"""
=============================================================
Custom collate function, Patients have different numbers of regions.

Example : Patient A (248,16) ,Patient B (263,16), Patient C (271,16)

We pad them inside ONE batch. Output (batch_size, max_regions_in_batch, feature_dim)

=============================================================
"""

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from data.region_dataset import RegionDataset

def region_collate(batch):
    """
    Custom collate function for RegionDataset. Pads the regions of each patient in the batch to the same length.
    we give in batch and its list
     [
        (patient_tensor,label),
        (patient_tensor,label),
        ...
    ]
    """
    region_tensor , labels = zip(*batch)  # Unzip the batch into separate lists

     # Save the ORIGINAL number of region n Example # tensor([248,263,271]) # We need this later so the 
     # model knows which rows are real.

    original_lengths = torch.tensor( [x.shape[0] for x in region_tensor],dtype=torch.long )  # Get the number of regions for each patient
    
    # Pad patients to the largest patient in this batch

    padded_regions = pad_sequence(region_tensor ,batch_first=True , padding_value=0.0)  # Pad the regions to the same length

    labels = torch.tensor(labels)

    return padded_regions, original_lengths ,labels



# if __name__ == "__main__":
#     dataset = RegionDataset(base_dir="src/data/raw/synthetic_cfdna_output")
    
#     loader = DataLoader(dataset , batch_size=4  ,shuffle=False , collate_fn=region_collate)

#     regions, original_lengths, labels = next(iter(loader)) 
    
#     print("First region of first patient:")
#     print(regions[0][0])  # Added an extra [0]

#     print("Batch shape :", regions.shape) #torch.Size([4, 264, 16])
#     print("Lengths :", original_lengths)
#     print("Labels :", labels)
