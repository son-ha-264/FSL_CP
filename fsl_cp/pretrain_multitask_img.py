import os
import json
import argparse

from utils.metrics import multitask_bce

import pytorch_lightning as pl
import torch


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
    args = parser.parse_args()

    image_path = args.path_to_image
    accelerator = args.device


    ### Inits
    dev_run = False
    version = 0
    fraction_train_set = 0.8
    loss_function = multitask_bce()

    ### Path inits
    HOME = os.environ['HOME']
    data_folder = os.path.join(HOME, 'FSL_CP/data/output')
    hparam_file = os.path.join(HOME, 'FSL_CP/fsl_cp/hparams/cnn_multitask.json')
    logs_path = os.path.join(HOME, 'FSL_CP/logs/multitask_img_pretrain')
    

    ### CHANGE WHICH GPU YOU WANT TO USE HERE 
    if accelerator == 'gpu':
        devices = [1,2,3]
    if accelerator == 'cpu':
        devices = 4
    

    ### Load hparams from JSON files
    with open(hparam_file) as f:
            hparams = json.load(f)

    
    ### Load data
    with open(os.path.join(data_folder, 'data_split.json')) as f:
        data = json.load(f)
    train_val_split = data['train'] + data['val']
