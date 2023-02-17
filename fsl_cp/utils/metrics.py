import torch
import numpy as np
from sklearn.metrics import average_precision_score


def _multitask_bce(pred, target):
    """Function to calculate multitask bce with missing data as -1. 
    Mask out -1, then average bce across tasks"""
    eps = 1e-7
    mask = (target != -1).float().detach() # not-1->1, -1->0
    bce = pred.clamp(min=0) - pred*target + torch.log(1.0 + torch.exp(-pred.abs()))
    bce[mask == 0] = 0
    loss = bce.sum() / (mask.sum() + eps)
    return(loss)


class multitask_bce(torch.nn.Module):
    """Class wrapper of the _multitask_bce function"""
    def __init__(self):
        super(multitask_bce, self).__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return(_multitask_bce(pred, target))


def accuracy_na(prediction, target):
    """Computes the precision when there is missing value in the data
    Missing values are denoted as -1"""
    mask = (target != -1)
    acc = ((target == prediction.round()) * mask).sum() / mask.sum()
    return acc


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def delta_auprc(true, pred):
    """Wrapper around sklearn's average_precision_score
    Delta AUPRC = average_precision_score - ratio of positives in true"""

    if type(true) != np.array:
        true = np.array(true)
    if type(pred) != np.array:
        pred = np.array(pred)
    
    auprc = average_precision_score(true, pred)
    baseline = np.sum(true)/len(true)

    return(auprc-baseline)

