import yaml
import torch

from data.data_loader import load_and_preprocess_data

from trainers.train_dl import (
    build_model,
    train_dl_model
)

from trainers.evaluate import evaluate_dl_model

from utils.seed_utils import set_seed


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def run_experiment(config_path, seed):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(seed)

    print(f"\nRunning Seed : {seed}")

    print(f"Using Device : {device}")

    data = load_and_preprocess_data(config,seed=seed)

    train_loader, val_loader, test_loader = data["loaders"]

    model = build_model(config, device)

    model = train_dl_model(
        model=model,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader
    )

    evaluate_dl_model(
        model,
        test_loader,
        device
    )