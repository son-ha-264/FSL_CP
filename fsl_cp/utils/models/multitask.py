from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.optim as optim


class cp_multitask(LightningModule):
    """LightningModule for organising all the training and validating steps
    """
    def __init__(self, model, loss_function,
                lr=1e-3, momentum=0.9, weight_decay=1e-4,
                step_size=20, gamma=0.1, num_classes=0):
        super().__init__()
        self.register_buffer("sigma", torch.eye(3))
        self.model = model
        self.save_hyperparameters(ignore=['loss_function', 'model'])
        self.loss_function = loss_function
        self.lr = lr
        self.momentum=momentum
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

    def accuracy(self, prediction, target):
        """Computes the precision@k for the specified values of k"""
        mask = (target != -1)
        acc = ((target == prediction.round()) * mask).sum() / mask.sum()
        return acc

    def forward(self, x):
        output = self.model(x)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat,y)
        sigmoid = nn.Sigmoid()
        acc = self.accuracy(sigmoid(y_hat), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_function(preds,labels)
        sigmoid = nn.Sigmoid()
        acc = self.accuracy(sigmoid(preds), labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        return preds
        
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]