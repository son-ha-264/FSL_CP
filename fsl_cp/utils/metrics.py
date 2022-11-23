import torch


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


def accuracy(prediction, target):
    """Computes the precision when there is missing value in the data
    Missing values are denoted as -1"""
    mask = (target != -1)
    acc = ((target == prediction.round()) * mask).sum() / mask.sum()
    return acc


