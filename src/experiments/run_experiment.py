import yaml
import torch

from imblearn.over_sampling import SMOTE

from data.balanced_data import balanced_data

from trainers.train_dl import (
    build_model,
    train_dl_model
)
from trainers.train_xgb import train_xgboost_model

from trainers.evaluate import evaluate_dl_model , evaluate_xgboost_model

from utils.seed_utils import set_seed
from sklearn.feature_selection import mutual_info_classif
import numpy as np

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def select_features_mi(X_train, y_train, X_val, X_test, k=10, random_state=42):
    """
    Select top k features using mutual information.
    Returns transformed arrays and the list of selected feature indices.
    """
    mi = mutual_info_classif(X_train, y_train, random_state=random_state)
    top_indices = np.argsort(mi)[::-1][:k]
    return X_train[:, top_indices], X_val[:, top_indices], X_test[:, top_indices], top_indices

def run_experiment(config_path, seed):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(seed)
    print(f"\nRunning Seed : {seed}")
    
    data = balanced_data(config, seed)
    model_type = config["model"]["type"]
    print(f"\nModel Type : {model_type}")
    num_features =config["dataset"]["num_features"]
    print(f"Number of features: {num_features}")
    file_path =config["dataset"]["filepath"]
    print(f"Data File Path: {file_path}")


    if model_type == "xgboost":
        print("Using Device : CPU/Native (XGBoost)")
        X_train, y_train, X_val, y_val, X_test, y_test = data["arrays"]
        
        # Mutual Information feature selection
        if config["models"]["xgboost"].get("mi_select", False):
            k = config["models"]["xgboost"].get("mi_k", 10)
            X_train, X_val, X_test, selected_indices = select_features_mi(
                X_train, y_train, X_val, X_test, k=k, random_state=seed
            )
            print(f"Selected {k} features by mutual information. Indices: {selected_indices}")
        
        model = build_model(config, device=None)
        model = train_xgboost_model(model, config, X_train, y_train, X_val, y_val)
        evaluate_xgboost_model(model, X_test, y_test)

    else:
        print(f"Using Device : {device}")
        train_loader, val_loader, test_loader = data["loaders"]
        # Optionally, if you want MI selection for DL models too, you can apply it to the arrays and rebuild loaders.
        # That's not implemented here, but you can add.
        model = build_model(config, device)
        model, class_weights_tensor = train_dl_model(
            model=model, config=config, device=device,
            train_loader=train_loader, val_loader=val_loader
        )
        evaluate_dl_model(model, test_loader, device, class_weights_tensor=class_weights_tensor)