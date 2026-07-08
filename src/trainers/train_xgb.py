import os
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

def tune_xgboost(X_train, y_train, config, n_trials=30):
    """
    Run Optuna hyperparameter search for XGBoost.
    Returns best parameters.
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 2),
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'random_state': config['project']['seed'][0],
        }
        # Use cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config['project']['seed'][0])
        model = xgb.XGBClassifier(**params)
        score = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_log_loss', n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params

def xgboost_multi_focal_loss(target_indices, logits, class_weights=None, gamma=2.0):
    """
    Computes numerically stable Softmax Focal Loss gradients and hessians for XGBoost.
    Uses the robust expectation approximation to prevent minority class tree clipping.
    """
    n_classes = 4  # Matches your dataset
    logits = logits.reshape(-1, n_classes)
    
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    n_samples, _ = probs.shape
    
    # One-hot encode targets safely
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n_samples), target_indices.astype(int)] = 1.0
    
    # Probability of the correct class per sample
    pt = np.sum(y_onehot * probs, axis=1, keepdims=True) 
    
    # ─── STABILIZED FOCAL STEP ───
    # Standard Softmax Cross-Entropy derivatives
    base_grad = probs - y_onehot
    base_hess = probs * (1.0 - probs)
    
    # Compute the modulating focal factor: (1 - pt)^gamma
    focal_weight = (1.0 - pt) ** gamma
    
    # Scale both to maintain stable, positive Hessian steps
    grad = base_grad * focal_weight
    hess = base_hess * focal_weight
    
    # Apply Class Weights if provided
    if class_weights is not None:
        weight_mask = np.array([class_weights[int(c)] for c in target_indices]).reshape(-1, 1)
        grad *= weight_mask
        hess *= weight_mask

    # Ensure hessian values don't hit zero absolute floor to avoid tree-split stalls
    hess = np.maximum(hess, 1e-4)

    return grad, hess

def xgboost_multi_focal_eval(y_true, y_pred):
    """
    Validation metric function matching the focal loss objective.
    """
    n_classes = 4  
    preds = y_pred.reshape(-1, n_classes)
    
    max_logits = np.max(preds, axis=1, keepdims=True)
    exp_logits = np.exp(preds - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    n_samples = probs.shape[0]
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n_samples), y_true.astype(int)] = 1.0
    
    pt = np.sum(y_onehot * probs, axis=1, keepdims=True)
    gamma = 2.0
    
    focal_loss_val = -np.mean(((1.0 - pt) ** gamma) * np.log(np.clip(pt, 1e-15, 1.0)))
    
    return focal_loss_val


def train_xgboost_model(model, config, X_train, y_train, X_val, y_val):
    print("\n" + "="*70)
    print("TRAINING XGBOOST MODEL (SMOTE BALANCED LOGIC)")
    print("="*70)
    
    xgb_cfg = config["models"]["xgboost"]
    
    # --- Optional: Hyperparameter Tuning ---
    if xgb_cfg.get("tune", False):
        print("Running Optuna hyperparameter tuning...")
        best_params = tune_xgboost(X_train, y_train, config, n_trials=xgb_cfg.get("n_trials", 30))
        # Update model with best params
        model.set_params(**best_params)
        # Also update the xgb_cfg for later reference (optional)
        xgb_cfg.update(best_params)
    else:
        # Use parameters from config (already set in build_model)
        # But we may still need to apply them if not using tune
        pass
    
    # ─── UPDATED: NO MANUAL WEIGHT COEFFICIENTS ───
    # Since the input dataset is balanced, class_weights passes to the objective as None.
    # We also keep gamma at 2.0 to balance standard sample difficulty tracking.
    def objective_fn(*args):
        arg1, arg2 = args[0], args[1]
        if len(arg1.shape) == 1 and arg1.shape[0] == len(y_train):
            return xgboost_multi_focal_loss(arg1, arg2, class_weights=None, gamma=2.0)
        else:
            return xgboost_multi_focal_loss(arg2, arg1, class_weights=None, gamma=2.0)

    # ─── HYPERPARAMETERS ───
    # Reverting to stable tree restrictions now that the small class leaves are filled with data
    xgb_cfg = config["models"]["xgboost"]
    model.set_params(
        objective=objective_fn,
        eval_metric=xgboost_multi_focal_eval,
        max_depth=xgb_cfg.get("max_depth", 6),
        min_child_weight=1.0,
        learning_rate=xgb_cfg.get("learning_rate", 0.1)
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Reset structural params alongside objectives prior to pickling
    model.set_params(objective="multi:softprob", eval_metric="mlogloss")
    
    artifacts_dir = config["project"].get("artifacts_dir", "./outputs")
    os.makedirs(artifacts_dir, exist_ok=True)
    save_path = os.path.join(artifacts_dir, f"xgboost_best.pkl")
    
    joblib.dump(model, save_path)
    print(f"\n--- Training Complete. Best model saved to {save_path}")
    
    return model