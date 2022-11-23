import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
import json, codecs
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.models.protonet import ProtoNet, FNN_Relu
from datamodule.protonet_cp import protonet_cp_dataset, protonet_cp_sampler
from utils.misc import sliding_average
import os
import pandas as pd


def fit(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        criterion, 
        optimizer,
        model
    ) -> float:
    """ Fit function for pretraining protonet. 
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
    if save_path:
        np.save(save_path, np.array(scores))
    return np.mean(scores), np.std(scores)


def main():
    ### Inits
    support_set_sizes = [16, 32, 64, 96]
    query_set_size = 32
    num_episodes_train = 2048
    num_episodes_test = 100

    json_path = '/home/son.ha/FSL_CP/data/output/data_split.json'
    label_df_path= '/home/son.ha/FSL_CP/data/output/FINAL_LABEL_DF.csv'
    cp_f_path=['/home/son.ha/FSL_CP/data/output/norm_CP_feature_df.csv']
    temp_folder = '/home/son.ha/FSL_CP/temp/np_files'
    df_assay_id_map_path = "/home/son.ha/FSL_CP/data/output/assay_target_map.csv"
    result_summary_path = '/home/son.ha/FSL_CP/result/result_summary/protonet_cp_result_summary.csv'

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
        print(f"Analysing for support set size {support_set_size}")
        for test_assay in tqdm(test_split):
            # Load data
            train_data = protonet_cp_dataset(
                train_split, 
                label_df_path= label_df_path, 
                cp_f_path=cp_f_path
            )
            test_data = protonet_cp_dataset(
                test_split, 
                label_df_path= label_df_path, 
                cp_f_path=cp_f_path
            )
            train_sampler = protonet_cp_sampler(
                task_dataset=train_data,
                support_set_size=support_set_size,
                query_set_size=query_set_size,
                num_episodes=num_episodes_train,
            )
            train_loader = DataLoader(
                train_data,
                batch_sampler=train_sampler,
                num_workers=12,
                pin_memory=True,
                collate_fn=train_sampler.episodic_collate_fn,
            )
            test_sampler = protonet_cp_sampler(
                task_dataset=test_data,
                support_set_size=support_set_size,
                query_set_size=query_set_size,
                num_episodes=num_episodes_test,
                specific_assay=test_assay 
            )
            test_loader = DataLoader(
                test_data,
                batch_sampler=test_sampler,
                num_workers=12,
                pin_memory=True,
                collate_fn=test_sampler.episodic_collate_fn,
            )
            # Load model
            input_shape=len(train_data[3][0])
            backbone = FNN_Relu(num_classes=1600, input_shape=input_shape)
            model = ProtoNet(backbone).cuda()

            # Performance before pretraining
            before_mean, before_std = evaluate(
                model, 
                test_loader, 
                save_path=os.path.join(temp_folder, f"protonet_before_{support_set_size}_{test_assay}.npy")
            )
            result_before_pretrain[str(support_set_size)+'_auc_before_train'].append(f"{before_mean:.2f}+/-{before_std:.2f}")

            # Pretrain on random assays
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
            after_mean, after_std = evaluate(
                model, 
                test_loader, 
                save_path=os.path.join(temp_folder, f"protonet_after_{support_set_size}_{test_assay}.npy")
            )
            final_result[str(support_set_size)+'_auc_after_train'].append(f"{after_mean:.2f}+/-{after_std:.2f}")

    # Create result summary dataframe
    df_assay_id_map = pd.read_csv(df_assay_id_map_path)
    df_assay_id_map = df_assay_id_map.astype({'ASSAY_ID': str})
    df_score_before = pd.DataFrame(data=result_before_pretrain)
    df_score_after = pd.DataFrame(data=final_result)
    df_score = pd.concat([df_score_before, df_score_after], axis=1)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path, index=False)


if __name__ == '__main__':
    main()