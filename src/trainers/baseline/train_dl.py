import numpy as np 
import yaml
import torch 
import torch.nn as nn
import torch.optim as optim
from model.ffn import FFN
from model.cnn import CNN
from utils.seed_utils import set_seed
from data.data_loader import load_and_preprocess_data
import xgboost as xgb
import os
from sklearn.utils.class_weight import compute_class_weight
import trainers.focal_loss as focal_loss
def train_dl_model( model: nn.Module, config:yaml, device ,train_loader , val_loader  ):
    
    model_cfg = config["models"][config["model"]["type"]]
    '''
    # Class imbalance handling: Gather all labels
    y_train = []
    for _, y in train_loader:
        y_train.extend(y.numpy())
    y_train = np.array(y_train)
    
    # ─── UPDATED: SMOOTHED SQUARE ROOT CLASS WEIGHTS ───
    # Count occurrences of each class
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    
    # Calculate smoothed weights using square root mapping
    weights = np.sqrt(total_samples / class_counts)
    
    # Normalize weights so their average remains 1.0
    weights = weights / np.mean(weights)
    
    # Convert to PyTorch tensor
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    '''
    
    class_weights_tensor = None # remove if going back to class weights for focal loss. 
    # Loss function with adjusted weights
    criterion = focal_loss.PyTorchFocalLoss(weight=class_weights_tensor, gamma=2.0)

    #criterion = nn.CrossEntropyLoss()
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


    print("\n" + "="*70)
    print(f"{'EPOCH':<7} | {'TRAIN LOSS':<10} | {'VAL LOSS':<8} | {'DEAD NEURONS LAYER 1':<20} | {'DEAD NEURONS LAYER 2':<20}")
    print("="*70)

    for epoch in range(epochs):

        if hasattr(model ,'clear_epoch_metrics'): #resetting the dead neuron avg here
            model.clear_epoch_metrics()

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

        # Calculate epoch averages for dead neurons
        r1_dead = np.mean(model.relu1_dead_pcts) * 100 if hasattr(model, 'relu1_dead_pcts') and model.relu1_dead_pcts else 0.0
        r2_dead = np.mean(model.relu2_dead_pcts) * 100 if hasattr(model, 'relu2_dead_pcts') and model.relu2_dead_pcts else 0.0

        # Output a clean, single-row report for this epoch
        print(f"Epoch {epoch:<2} | {train_loss:<10.4f} | {val_loss:<8.4f} | {r1_dead:<18.2f}% | {r2_dead:<18.2f}%")
        

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            patience_counter =0 
            # Save the absolute best weights found so far (Checkpointing)
            
            artifacts_dir = config["project"].get("artifacts_dir", "./outputs")
            os.makedirs(artifacts_dir, exist_ok=True)
            save_path = os.path.join(artifacts_dir, f"{config['model']['type']}_{config['model']['save_path']}")
            #save_path = f"{config['model']['type']}_{config['model']['save_path']}"

            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1 # No improvement

        if patience_counter >= model_cfg["early_stopping_patience"]:
            print(f"Early Stopping triggered at epoch {epoch+1}! Validation loss stalled.")
            break
    print("\n--- Training Complete. Loading best checkpoint ")
    model.load_state_dict(torch.load(save_path))

    return model ,class_weights_tensor



def build_model(config, device):
    model_type = config["model"]["type"]

    if model_type == "ffn":
        return FFN(config).to(device)

    elif model_type == "cnn":
        return CNN(config).to(device)

    elif model_type == "xgboost":
            # Initialize the XGBoost Classifier using hyperparameters from config
            xgb_cfg = config["models"]["xgboost"]
            return xgb.XGBClassifier(
                max_depth=xgb_cfg["max_depth"],
                learning_rate=xgb_cfg["learning_rate"],
                n_estimators=xgb_cfg["n_estimators"],
                subsample=xgb_cfg["subsample"],
                colsample_bytree=xgb_cfg["colsample_bytree"],
                objective=None,  # ◄ CHANGED: Set to None so our Custom Focal Loss can take over smoothly
                early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
                max_delta_step=xgb_cfg.get("max_delta_step", 0), 
                # ◄ REMOVED: eval_metric="mlogloss" (Custom objectives manage raw margins directly)
                random_state=config["project"]["seed"][0]
            )

    else:
        raise ValueError(f"Unknown model type: {model_type}")






