import torch 
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import json
import timm
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score

from utils.metrics import delta_auprc
from utils.misc import NormalizeByImage
from utils.models.protonet import ProtoNet
from torch.optim.lr_scheduler import StepLR
from datamodule.protonet_img import protonet_img_dataset, protonet_img_sampler


def fit(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        criterion, 
        optimizer,
        model, 
        device
    ) -> float:
    """ Fit function for training protonet. 
    """
    optimizer.zero_grad()
    classification_scores = model(
        support_images.to(device), support_labels.to(device), query_images.to(device)
    )

    loss = criterion(classification_scores, query_labels.to(device))
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_on_one_task(
    model,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    device
):
    """
    Returns the prediction of the protonet model and the real label
    """
    return (
        torch.max(
            model(support_images.to(device), support_labels.to(device), query_images.to(device))
            .detach()
            .data,
            1,
        )[1]).cpu().numpy(), query_labels.cpu().numpy()


def evaluate(model, data_loader: DataLoader, device):
    """ Evaluate the model on a DataLoader object
    """
    AUROC_scores = []
    dAUPRC_scores = []
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in enumerate(data_loader):
            y_pred, y_true = evaluate_on_one_task(
                model, support_images, support_labels, query_images, query_labels, device
            )
            AUROC_score = roc_auc_score(y_true, y_pred)
            dAUPRC_score = delta_auprc(y_true, y_pred)
            AUROC_scores.append(AUROC_score)
            dAUPRC_scores.append(dAUPRC_score)

    return np.mean(AUROC_scores), np.std(AUROC_scores), np.mean(dAUPRC_scores), np.std(dAUPRC_scores)


def main(seed=69):

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--path_to_image', type=str, default='/mnt/scratch/Son_cellpainting/my_cp_images/',
        help='Path to folder of Cell Painting images')
    parser.add_argument(
        '-d', '--device', type=str, default='cuda:0',
        help='gpu(cuda) or cpu')
    args = parser.parse_args()

    image_path = args.path_to_image
    device = args.device

    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    ##### Inits
    # Random inits
    support_set_sizes = [8, 16, 32, 64, 96]
    query_set_size = 32
    num_episodes_train = 70000
    num_episodes_test = 100
    crop_size = 400
    image_resize = 200
    step_size = 20000

    # Path name inits
    HOME = os.environ['HOME']
    json_path = os.path.join(HOME, 'FSL_CP/data/output/data_split.json')
    label_df_path = os.path.join(HOME,'FSL_CP/data/output/FINAL_LABEL_DF.csv')
    #image_path = '/mnt/scratch/Son_cellpainting/my_cp_images/'
    image_path = image_path
    df_assay_id_map_path = os.path.join(HOME,'FSL_CP/data/output/assay_target_map.csv')
    result_summary_path1 = os.path.join(HOME,'FSL_CP/result/result_summary/protonet_img_auroc_result_summary.csv')
    result_summary_path2 = os.path.join(HOME,'FSL_CP/result/result_summary/protonet_img_dauprc_result_summary.csv')
    
    # Result dictionary init

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
         transforms.RandomCrop(crop_size),
        transforms.Resize((image_resize,image_resize)),
        NormalizeByImage()]) 


    ### Load the assay keys
    with open(json_path) as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']
 
    train_split = train_split + val_split
    final_result_auroc['ASSAY_ID'] = test_split
    final_result_dauprc['ASSAY_ID'] = test_split
    

    ### Loop through all support set size, performing few-shot prediction:
    for support_set_size in tqdm(support_set_sizes):
        tqdm.write(f"Analysing for support set size {support_set_size}")
        for test_assay in tqdm(test_split, desc='Test Assay Index', leave=False):
            # Load data
            #tqdm.write('Load data...')
            train_data = protonet_img_dataset(
                train_split, 
                label_df_path= label_df_path, 
                image_path=image_path,
                transform=transform
            )
            train_sampler = protonet_img_sampler(
                task_dataset=train_data,
                support_set_size=support_set_size,
                query_set_size=query_set_size,
                num_episodes=num_episodes_train,
            )
            train_loader = DataLoader(
                train_data,
                batch_sampler=train_sampler,
                num_workers=20,
                pin_memory=True,
                collate_fn=train_sampler.episodic_collate_fn,
            )
            test_data = protonet_img_dataset(
                test_split, 
                label_df_path= label_df_path, 
                image_path=image_path,
                transform=transform
            )
            test_sampler = protonet_img_sampler(
                task_dataset=test_data,
                support_set_size=support_set_size,
                query_set_size=query_set_size,
                num_episodes=num_episodes_test,
                specific_assay=test_assay 
            )
            test_loader = DataLoader(
                test_data,
                batch_sampler=test_sampler,
                num_workers=20,
                pin_memory=True,
                collate_fn=test_sampler.episodic_collate_fn,
            )
            # Load model
            #tqdm.write('Load model...')
            backbone = timm.create_model('resnet50', in_chans=5, pretrained=False)
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, 1600, bias=True) 
            model = ProtoNet(backbone).to(device)

            # Pretrain on random assays
            #tqdm.write('Pretrain...')
            criterion = nn.CrossEntropyLoss()
            #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.1)   
            all_loss = []
            model.train()
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in enumerate(train_loader):
                loss_value = fit(support_images, support_labels, query_images, query_labels, criterion, optimizer, model, device)
                if episode_index % step_size == 0:
                    scheduler.step()
                all_loss.append(loss_value)

            # Performance after pretraining
            #tqdm.write('Performance after...')
            auroc_mean, auroc_std, dauprc_mean, dauprc_std = evaluate(
                model, 
                test_loader, 
                device
            )
            final_result_auroc[str(support_set_size)].append(f"{auroc_mean:.2f}+/-{auroc_std:.2f}")
            final_result_dauprc[str(support_set_size)].append(f"{dauprc_mean:.2f}+/-{dauprc_std:.2f}")


    ### Create result summary dataframe
    df_assay_id_map = pd.read_csv(df_assay_id_map_path)
    df_assay_id_map = df_assay_id_map.astype({'ASSAY_ID': str})

    df_score = pd.DataFrame(data=final_result_auroc)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path1, index=False)

    df_score = pd.DataFrame(data=final_result_dauprc)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path2, index=False)


if __name__ == '__main__':
    main()