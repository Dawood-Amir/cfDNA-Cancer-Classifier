import torch
import torch.nn as nn

from utils.metrics import compute_metrics


def evaluate_dl_model(model:nn.Module, test_loader, device):

    model.eval()

    criterion = nn.CrossEntropyLoss()

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