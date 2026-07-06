from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from data.data_loader import load_and_preprocess_data

def balanced_data(config, seed):
    # Load original data splits
    data = load_and_preprocess_data(config, seed=seed)
    X_train, y_train, X_val, y_val, X_test, y_test = data["arrays"]
    train_loader, val_loader, test_loader = data["loaders"]

    TARGET_COUNT = 200

    print(f"\n--- Original Training Class Distribution: {np.bincount(y_train)}")

    # 1. First, down-sample the massive majority classes to 200
    under_strategy = {0: TARGET_COUNT, 1: TARGET_COUNT, 2: TARGET_COUNT}
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=seed)
    X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

    # 2. Next, up-sample the minority class (Class 3) to 200 using SMOTE
    over_strategy = {3: TARGET_COUNT}
    smote = SMOTE(sampling_strategy=over_strategy, random_state=seed)
    X_train_res, y_train_res = smote.fit_resample(X_train_under, y_train_under)
    
    print(f"--- Hybrid Balanced Training Class Distribution: {np.bincount(y_train_res)}\n")
    
    # 3. Rebuild the PyTorch DataLoader using the hybrid-balanced arrays
    balanced_dataset = TensorDataset(
        torch.tensor(X_train_res, dtype=torch.float32),
        torch.tensor(y_train_res, dtype=torch.long)
    )
    
    train_loader_bal = DataLoader(
        balanced_dataset, 
        batch_size=train_loader.batch_size, 
        shuffle=True
    )

    # Return structured containers containing the exact objects the experiment file expects
    return {
        "arrays": (X_train_res, y_train_res, X_val, y_val, X_test, y_test),
        "loaders": (train_loader_bal, val_loader, test_loader),
    }