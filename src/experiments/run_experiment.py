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


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def run_experiment(config_path, seed):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(seed)
    print(f"\nRunning Seed : {seed}")
    
    # Call your new function to get clean, hybrid balanced objects
    data = balanced_data(config, seed)
    model_type = config["model"]["type"]
    print(f"\nModel Type : {model_type}")

    if model_type == "xgboost":
        print("Using Device : CPU/Native (XGBoost)")
        X_train, y_train, X_val, y_val, X_test, y_test = data["arrays"]
        
        model = build_model(config, device=None)
        model = train_xgboost_model(model, config, X_train, y_train, X_val, y_val)
        evaluate_xgboost_model(model, X_test, y_test)

    else:
        print(f"Using Device : {device}")
        train_loader, val_loader, test_loader = data["loaders"]

        model = build_model(config, device)
        model, class_weights_tensor = train_dl_model(
            model=model, config=config, device=device,
            train_loader=train_loader, val_loader=val_loader
        )
        evaluate_dl_model(model, test_loader, device, class_weights_tensor=class_weights_tensor)