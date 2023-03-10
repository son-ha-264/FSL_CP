import os
import json
import timm
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score
from os.path import expanduser

from utils.misc import NormalizeByImage
from utils.metrics import multitask_bce, delta_auprc
from datamodule.multitask_img import each_view_a_datapoint, multitask_pretrain_img_dataset

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from utils.models.multitask import cp_multitask


def main(
    seed=69
):
    
    ### Seed
    pl.seed_everything(seed)
    torch.manual_seed(seed)


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--path_to_image', type=str, default='/mnt/scratch/Son_cellpainting/my_cp_images/',
        help='Path to folder of Cell Painting images')
    parser.add_argument(
        '-d', '--device', type=str, default='gpu',
        help='gpu or cpu')
    parser.add_argument(
        '-r', '--devrun', type=bool, default=False,
        help='dev run (faster run for debugging)')
    parser.add_argument(
        '-s', '--resize', type=int, default=520,
        help='Resize the shortest edge to this value. The longer edge is also scaled to keep the ratio')
    args = parser.parse_args()

    image_path = args.path_to_image
    dev_run = args.devrun
    accelerator = args.device
    image_resize = args.resize


    ### Inits
    dev_run = dev_run
    version = 0
    fraction_train_set = 0.8
    image_resize = image_resize
    loss_function = multitask_bce()

    ### Path inits
    HOME = expanduser("~")
    data_folder = os.path.join(HOME, 'FSL_CP/data/output')
    hparam_file = os.path.join(HOME, 'FSL_CP/fsl_cp/hparams/cnn_multitask.json')
    logs_path = os.path.join(HOME, 'FSL_CP/logs/multitask_img_pretrain')
    

    ### CHANGE WHICH GPU YOU WANT TO USE HERE 
    if accelerator == 'gpu':
        devices = [2,3]
    if accelerator == 'cpu':
        devices = 4
    

    ### Result dictionary init
    final_result_auroc = {
        '8': [],
        '16': [],
        '32': [],
        '64': [],
        '96': []
    }

    final_result_dauprc = {
        '8': [],
        '16': [],
        '32': [],
        '64': [],
        '96': []
    }

    ### Define image transformation
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(image_resize),
        NormalizeByImage()]) 
    

    ### Load hparams from JSON files
    with open(hparam_file) as f:
            hparams = json.load(f)

    
    ### Load data
    with open(os.path.join(data_folder, 'data_split.json')) as f:
        data = json.load(f)
    train_val_split = data['train'] + data['val']

    label_df = pd.read_csv(os.path.join(data_folder, 'FINAL_LABEL_DF.csv'))
    label_df = each_view_a_datapoint(label_df)

    train_val_data = multitask_pretrain_img_dataset(
         assay_codes=train_val_split,
         image_path=image_path,
         label_df=label_df,
         transform=transform
    )
    train_size = int(fraction_train_set * len(train_val_data))
    val_size = len(train_val_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_val_data, [train_size, val_size])

    ### Create DataLoader objects
    train_loader = DataLoader(train_data, batch_size=hparams["training"]["batch_size"],
                            shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=hparams["training"]["batch_size"],
                            shuffle=False, num_workers=8)
    

    # Load model
    model = timm.create_model('resnet50', in_chans=5, pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_data[3][1]), bias=True)


    ### Small hparams change for dev run
    if dev_run:
        max_epochs = 2
        limit_train_batches = 2
        limit_val_batches = 2
    else:
        max_epochs = hparams["training"]["max_epochs"]
        limit_train_batches = 1.0
        limit_val_batches = 1.0

    
    ### Model pretraining 
    tb_logger = pl_loggers.TensorBoardLogger(logs_path, version = version, default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    train_loop = cp_multitask(
        model, loss_function,
        lr=hparams["optimizer_params"]["lr"],
        momentum=hparams["optimizer_params"]["momentum"],
        weight_decay=hparams["optimizer_params"]["weight_decay"],
        step_size=hparams["lr_schedule"]["step_size"],
        gamma=hparams["lr_schedule"]["gamma"],
        num_classes=len(train_data[3][1])
    )
    checkpoint_callback = ModelCheckpoint(mode="min", monitor="val_loss", save_top_k = 1)
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=devices, 
        max_epochs=max_epochs,
        logger=tb_logger,
        gradient_clip_val=hparams["grad_clip"]["val"], 
        gradient_clip_algorithm=hparams["grad_clip"]["algorithm"],
        strategy=DDPStrategy(find_unused_parameters=False),
        limit_train_batches=limit_train_batches, 
        limit_val_batches=limit_val_batches, 
        log_every_n_steps=20,
        callbacks=[
            checkpoint_callback,
            lr_monitor
        ],
    )

    trainer.fit(train_loop, train_loader, val_loader)


    ### Save the weights of the last epoch
    trainer.save_checkpoint(
        os.path.join(
            logs_path,
            "lightning_logs/",
            "version_"+str(version),
            "checkpoints",
            "final_model.ckpt"
        )
    )

    return None

if __name__ == '__main__':
     main()