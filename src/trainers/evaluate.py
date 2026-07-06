from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np

from utils.metrics import compute_metrics
import trainers.focal_loss as focal_loss

def evaluate_dl_model(model:nn.Module, test_loader, device ,class_weights_tensor=None):

    model.eval()

    criterion = focal_loss.PyTorchFocalLoss(weight=class_weights_tensor, gamma=2.0)

    test_loss = 0

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            loss = criterion(logits, y)
            test_loss += loss.item() * X.size(0)

            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    metrics = compute_metrics(
        y_true,
        y_pred,
        y_probs
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Loss : {test_loss:.4f}")
    print(f"AUC  : {metrics['auc']:.4f}")
    print(metrics["classification_report"])
    print(metrics["confusion_matrix"])

    return metrics


def evaluate_xgboost_model(model: xgb.XGBClassifier, X_test, y_test):
    # ─── UPDATED: Extract raw margin margins instead of default softprob ───
    # For custom objectives, predict() outputs raw margins (logits)
    # shape: [n_samples, n_classes]
    logits = model.predict(X_test)
    
    # If the model output format varies by version, ensure it's treated as margins
    if len(logits.shape) == 1 or logits.shape[1] != 4:
        # Fallback check if your model version treats predict differently with custom objectives
        logits = model.get_booster().predict(xgb.DMatrix(X_test), output_margin=True)

    # Apply manual Softmax to convert raw focal margins back to clean probabilities
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    y_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Assign discrete class assignments using the highest probability index
    y_pred = np.argmax(y_probs, axis=1)
    
    test_loss = log_loss(y_test, y_probs)

    # Compute metrics using your existing compute_metrics helper
    metrics = compute_metrics(
        y_true=list(y_test),
        y_pred=list(y_pred),
        y_probs=list(y_probs)
    )
    
    print("\n========== XGBOOST FOCAL TEST RESULTS ==========")
    print(f"Loss : {test_loss:.4f}")
    print(f"AUC  : {metrics['auc']:.4f}")
    print(metrics["classification_report"])
    print(metrics["confusion_matrix"])
    
    return metrics