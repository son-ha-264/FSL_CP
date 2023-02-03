import torch 
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import json
import timm
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score

from utils.misc import NormalizeByImage
from utils.models.protonet import ProtoNet
from datamodule.protonet_img import protonet_img_dataset, protonet_img_sampler


def fit(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        criterion, 
        optimizer,
        model
    ) -> float:
    """ Fit function for training protonet. 
    """
    optimizer.zero_grad()
    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )

    loss = criterion(classification_scores, query_labels.cuda())
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_on_one_task(
    model,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
):
    """
    Returns the prediction of the protonet model and the real label
    """
    return (
        torch.max(
            model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
            .detach()
            .data,
            1,
        )[1]).cpu().numpy(), query_labels.cpu().numpy()


def evaluate(model, data_loader: DataLoader, save_path=None):
    """ Evaluate the model on a DataLoader object
    """
    scores = []
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
                model, support_images, support_labels, query_images, query_labels
            )
            score = roc_auc_score(y_true, y_pred)
            scores.append(score)

    # (optional) Save scores to a NumPy file
    if save_path:
        np.save(save_path, np.array(scores))
    return np.mean(scores), np.std(scores)


def main():

    ##### Inits

    # Random inits
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    support_set_sizes = [16, 32, 64, 96]
    query_set_size = 32
    num_episodes_train = 8#2048
    num_episodes_test = 100
    image_resize = 20#80

    # Path name inits
    json_path = '../data/output/data_split.json'
    label_df_path = '../data/output/FINAL_LABEL_DF.csv'
    image_path = '/mnt/scratch/Son_cellpainting/my_cp_images/'
    temp_folder = '../temp/np_files'
    df_assay_id_map_path = "../data/output/assay_target_map.csv"
    result_summary_path = '../result/result_summary/protonet_img_result_summary.csv'
    
    # Result dictionary init
    result_before_pretrain = {
        'ASSAY_ID': [],
        '16_auc_before_train': [],
        '32_auc_before_train': [],
        '64_auc_before_train': [],
        '96_auc_before_train': []
    }
    final_result = {
        '16_auc_after_train': [],
        '32_auc_after_train': [],
        '64_auc_after_train': [],
        '96_auc_after_train': []
    }


    ### Define image transformation
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((image_resize,image_resize)),
        NormalizeByImage()]) 


    ### Load the assay keys
    with open(json_path) as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']
 
    train_split = train_split + val_split
    result_before_pretrain['ASSAY_ID'] = test_split
    

    ### Loop through all support set size, performing few-shot prediction:
    for support_set_size in support_set_sizes:
        tqdm.write(f"Analysing for support set size {support_set_size}")
        for test_assay in tqdm(test_split, desc='Test Assay Index'):
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
            model = ProtoNet(backbone).cuda()

            # Performance before pretraining
            #tqdm.write('Performance before...')
            before_mean, before_std = evaluate(
                model, 
                test_loader, 
                #save_path=os.path.join(temp_folder, f"protonet_before_{support_set_size}_{test_assay}.npy")
            )
            result_before_pretrain[str(support_set_size)+'_auc_before_train'].append(f"{before_mean:.2f}+/-{before_std:.2f}")

            # Pretrain on random assays
            #tqdm.write('Pretrain...')
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            all_loss = []
            model.train()
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in enumerate(train_loader):
                loss_value = fit(support_images, support_labels, query_images, query_labels, criterion, optimizer, model)
                all_loss.append(loss_value)

            # Performance after pretraining
            #tqdm.write('Performance after...')
            after_mean, after_std = evaluate(
                model, 
                test_loader, 
                #save_path=os.path.join(temp_folder, f"protonet_after_{support_set_size}_{test_assay}.npy")
            )
            final_result[str(support_set_size)+'_auc_after_train'].append(f"{after_mean:.2f}+/-{after_std:.2f}")


    ### Create result summary dataframe
    df_assay_id_map = pd.read_csv(df_assay_id_map_path)
    df_assay_id_map = df_assay_id_map.astype({'ASSAY_ID': str})
    df_score_before = pd.DataFrame(data=result_before_pretrain)
    df_score_after = pd.DataFrame(data=final_result)
    df_score = pd.concat([df_score_before, df_score_after], axis=1)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path, index=False)


if __name__ == '__main__':
    main()