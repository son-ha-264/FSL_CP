from datamodule.multitask_cp import multitask_pretrain_cp_dataset
from utils.models.multitask import cp_multitask
from utils.models.shared_models import FNN_Relu
from utils.metrics import multitask_bce
import json
import pytorch_lightning as pl
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

def main(
        seed=69
        ):

    ### Seed
    pl.seed_everything(69)
    torch.manual_seed(seed)

    ### Inits
    dev_run = False
    version = 0
    fraction_train_set = 0.8
    devices = [3]
    loss_function = multitask_bce()


    ### Path inits 
    data_folder = '/home/son.ha/FSL_CP/data/output'
    hparam_file = '/home/son.ha/FSL_CP/fsl_cp/hparams/fnn_multitask.json'
    cp_f_path = os.path.join(data_folder, 'norm_CP_feature_df.csv') 
    feature_df = pd.read_csv(cp_f_path)
    logs_path = '/home/son.ha/FSL_CP/logs/multitask_only_cp_pretrain'


    ### Load hparams from JSON files
    with open(hparam_file) as f:
            hparams = json.load(f)


    ### Load data
    with open(os.path.join(data_folder, 'data_split.json')) as f:
        data = json.load(f)
    train_val_split = data['train'] + data['val']

    train_val_data = multitask_pretrain_cp_dataset(
        assay_codes=train_val_split,
        label_df_path=os.path.join(data_folder, 'FINAL_LABEL_DF.csv'),
        feature_df=feature_df,
    )

    train_size = int(fraction_train_set * len(train_val_data))
    val_size = len(train_val_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_val_data, [train_size, val_size])


    ### Create DataLoader objects
    train_loader = DataLoader(train_data, batch_size=hparams["training"]["batch_size"],
                            shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=hparams["training"]["batch_size"],
                            shuffle=False, num_workers=8)


    ### Models
    model = FNN_Relu(input_shape=len(train_data[3][0]), num_classes=len(train_data[3][1]))


    ### Small hparams change for dev run
    if dev_run:
        max_epochs = 2
        limit_train_batches = 3
        limit_val_batches = 3
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