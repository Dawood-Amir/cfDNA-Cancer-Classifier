import numpy as np 
import yaml
import torch 
import torch.nn as nn
import torch.optim as optim
from model.ffn import FFN
from model.cnn import CNN
from utils.seed_utils import set_seed
from data.data_loader import load_and_preprocess_data


def train_dl_model( model: nn.Module, config:yaml, device ,train_loader , val_loader  ):
    
    model_cfg = config["models"][config["model"]["type"]]

    #fetch data channels

    #lossfunc
    criterion = nn.CrossEntropyLoss()

    #optimization
    optimizer  = optim.AdamW(
        model.parameters() ,lr=model_cfg['learning_rate'],
          weight_decay=model_cfg['weight_decay']
        )
    
    #Learning Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer= optimizer ,
        mode="min",
        factor=model_cfg["lr_scheduler_factor"] ,
        patience=model_cfg["lr_scheduler_patience"]  
    )
    epochs = model_cfg["epochs"]

    best_val_loss =float('inf') 
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X,y in train_loader:
            X,y = X.to(device), y.to(device)

            optimizer.zero_grad()
            out =model(X)

            loss = criterion(out,y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)

        train_loss  = train_loss/len(train_loader.dataset)

        # -------------------
        # VALIDATION PHASE
        # -------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                avg_loss = criterion(out, y)
                val_loss += avg_loss.item() * X.size(0)
        val_loss/=len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            patience_counter =0 #
            # Save the absolute best weights found so far (Checkpointing)
            torch.save(model.state_dict(), config["model"]["save_path"])

        else:
            patience_counter += 1 # No improvement, increment panic counter 

        if patience_counter >= model_cfg["early_stopping_patience"]:
            print(f"Early Stopping triggered at epoch {epoch+1}! Validation loss stalled.")
            break
    print("\n--- Training Complete. Loading best checkpoint ")
    model.load_state_dict(torch.load(config["model"]["save_path"]))

    return model



def build_model(config, device):
    model_type = config["model"]["type"]

    if model_type == "ffn":
        return FFN(config).to(device)

    elif model_type == "cnn":
        return CNN(config).to(device)

    elif model_type == "xgboost":
        return None  # handled separately

    else:
        raise ValueError(f"Unknown model type: {model_type}")






