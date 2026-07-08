import torch
import torch.nn as nn

class PyTorchFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(PyTorchFocalLoss, self).__init__()
        self.weight = weight # Handled class imbalance tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute standard cross entropy loss without reducing it yet
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        
        # Compute probability of the correct class
        pt = torch.exp(-ce_loss)
        
        # Calculate Focal Loss adjustment
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss