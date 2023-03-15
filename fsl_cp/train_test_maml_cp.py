import os
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import learn2learn as l2l
from os.path import expanduser
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, accuracy_score, roc_auc_score

from utils.metrics import delta_auprc, accuracy, multitask_bce
from utils.models.shared_models import FNN_Relu
from datamodule.maml_cp import maml_img_dataset, maml_cp_sampler

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_miniimagenet.py

def fast_adapt(
        support_f: torch.Tensor,
        support_labels: torch.Tensor,
        query_f: torch.Tensor,
        query_labels: torch.Tensor,
        loss, 
        learner, 
        adaptation_steps, 
        device
        ):
    # Move to GPU
    support_f = support_f.to(device)
    support_labels = support_labels.to(device)
    query_f = query_f.to(device)
    query_labels = query_labels.to(device)

    # Adapt model
    for step in range(adaptation_steps):
        adaptation_error = loss(torch.squeeze(learner(support_f)), support_labels.float())
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(query_f)
    evaluation_error = loss(torch.squeeze(predictions), query_labels.float())
    #evaluation_accuracy = accuracy(torch.squeeze(predictions), query_labels.float())
    return evaluation_error#, evaluation_AUROC, evaluation_dAUPRC


def evaluate(original_maml, data_loader, device, adaptation_steps, loss):
    """ Evaluate the model on a DataLoader object
    """
    AUROC_scores = []
    dAUPRC_scores = []
    bacc_scores = []
    F1_scores = []
    kappa_scores = []
    for episode_index, (
        support_f,
        support_labels,
        query_f,
        query_labels,
        class_ids,
    ) in enumerate(data_loader):

        learner = original_maml.clone()

        # Move to GPU
        support_f = support_f.to(device)
        support_labels = support_labels.to(device)
        query_f = query_f.to(device)

        # Adapt model
        for step in range(adaptation_steps):
            adaptation_error = loss(torch.squeeze(learner(support_f)), support_labels.float())
            learner.adapt(adaptation_error)

        # Evaluate the adapted model
        predictions = learner(query_f)
        s = torch.nn.Sigmoid()
        predictions = s(predictions).detach().cpu().numpy()
        pred_round = np.rint(predictions)
        AUROC_score = roc_auc_score(query_labels, predictions)
        dAUPRC_score = delta_auprc(query_labels, predictions)
        bacc_score = balanced_accuracy_score(query_labels, pred_round, adjusted=True)
        F1_score = f1_score(query_labels, pred_round)
        kappa_score = cohen_kappa_score(query_labels, pred_round)

        AUROC_scores.append(AUROC_score)
        dAUPRC_scores.append(dAUPRC_score)
        bacc_scores.append(bacc_score)
        F1_scores.append(F1_score)
        kappa_scores.append(kappa_score)
    return np.mean(AUROC_scores), np.std(AUROC_scores), np.mean(dAUPRC_scores), np.std(dAUPRC_scores), np.mean(bacc_scores), np.std(bacc_scores), np.mean(F1_scores), np.std(F1_scores), np.mean(kappa_scores), np.std(kappa_scores)
           


def main(
    support_set_sizes = [8, 16, 32, 64, 96],
    query_set_size = 32,
    num_episodes_train = 1000,
    num_episodes_test = 100,
    meta_batch_size=32,
    adaptation_steps=3,
    cuda=True,
    seed=69
    ):

    ### Seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--device', type=str, default='cuda:0',
        help='gpu(cuda) or cpu')
    args = parser.parse_args()

    device = args.device
    cuda_device=device
    
    ### Inits
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        #torch.cuda.manual_seed(seed)
        device = torch.device(cuda_device)

    ### Paths inits
    HOME = expanduser("~")
    df_assay_id_map_path = os.path.join(HOME, 'FSL_CP/data/output/assay_target_map.csv') 
    json_path = os.path.join(HOME, 'FSL_CP/data/output/data_split.json') 
    label_df_path= os.path.join(HOME, 'FSL_CP/data/output/FINAL_LABEL_DF.csv')
    #cp_f_path=[os.path.join(HOME, 'FSL_CP/data/output/norm_CP_feature_df.csv')]
    cp_f_path=[os.path.join(HOME,'FSL_CP/data/output/norm_CP_feature_df.csv'),
               os.path.join(HOME,'FSL_CP/data/output/cnn_embeddings.csv')]
    
    feature = 'cp+'
    result_summary_path1 = os.path.join(HOME, f"FSL_CP/result/result_summary2/maml_{feature}_auroc_result_summary.csv") 
    result_summary_path2 = os.path.join(HOME, f"FSL_CP/result/result_summary2/maml_{feature}_dauprc_result_summary.csv") 
    result_summary_path3 = os.path.join(HOME, f"FSL_CP/result/result_summary2/maml_{feature}_bacc_result_summary.csv") 
    result_summary_path4 = os.path.join(HOME, f"FSL_CP/result/result_summary2/maml_{feature}_f1_result_summary.csv") 
    result_summary_path5 = os.path.join(HOME, f"FSL_CP/result/result_summary2/maml_{feature}_kappa_result_summary.csv") 

    ### Final result dictionary
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

    final_result_bacc = {
        '8': [],
        '16': [],
        '32': [],
        '64': [],
        '96': []
    }

    final_result_f1 = {
        '8': [],
        '16': [],
        '32': [],
        '64': [],
        '96': []
    }

    final_result_kappa = {
        '8': [],
        '16': [],
        '32': [],
        '64': [],
        '96': []
    }

    ### Load the assay keys
    with open(json_path) as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']
 
    final_result_auroc['ASSAY_ID'] = test_split
    final_result_dauprc['ASSAY_ID'] = test_split
    final_result_bacc['ASSAY_ID'] = test_split
    final_result_f1['ASSAY_ID'] = test_split
    final_result_kappa['ASSAY_ID'] = test_split


    for support_set_size in support_set_sizes:
        tqdm.write(f"Analysing for support set size {support_set_size}")

        # Load train data
        train_data = maml_img_dataset(
            train_split, 
            label_df_path= label_df_path, 
            cp_f_path=cp_f_path
        )
        train_sampler = maml_cp_sampler(
                task_dataset=train_data,
                support_set_size=support_set_size,
                query_set_size=query_set_size,
                num_episodes=num_episodes_train,
                meta_batch_size=meta_batch_size
        )
        train_loader = DataLoader(
                train_data,
                batch_sampler=train_sampler,
                num_workers=12,
                pin_memory=True,
                collate_fn=train_sampler.episodic_collate_fn,
        )

        # Model
        input_shape=len(train_data[3][0])
        model= FNN_Relu(num_classes=1, input_shape=input_shape)
        model.to(device)
        maml = l2l.algorithms.MAML(model, lr=0.01, first_order=False) 
        opt = optim.Adam(maml.parameters(), 0.001)
        loss = nn.BCEWithLogitsLoss()

        # Train
        for episode_index, (
            support_f,
            support_labels,
            query_f,
            query_labels,
            _,
        ) in tqdm(enumerate(train_loader), desc='Train model', leave=False, total=num_episodes_train*meta_batch_size):
            if episode_index % meta_batch_size == 0:
                opt.zero_grad()
                meta_train_error = 0.0

            learner = maml.clone()
            evaluation_error = fast_adapt(
                support_f=support_f,
                support_labels=support_labels,
                query_f=query_f,
                query_labels=query_labels,
                loss=loss, 
                learner=learner, 
                adaptation_steps=adaptation_steps, 
                device=device
            )

            evaluation_error.backward()
            meta_train_error += evaluation_error.item()

            # Average the accumulated gradients and optimize
            if (episode_index+1) % meta_batch_size == 0:
                for p in maml.parameters():
                    p.grad.data.mul_(1.0 / meta_batch_size)
                opt.step()


        for test_assay in tqdm(test_split):
            test_assays = [test_assay]
            test_data = maml_img_dataset(
                    test_assays, 
                    label_df_path=label_df_path, 
                    cp_f_path=cp_f_path
            )
            test_sampler = maml_cp_sampler(
                    task_dataset=test_data,
                    support_set_size=support_set_size,
                    query_set_size=query_set_size,
                    num_episodes=num_episodes_test,
                    meta_batch_size=meta_batch_size
            )
            test_loader = DataLoader(
                    test_data,
                    batch_sampler=test_sampler,
                    num_workers=12,
                    pin_memory=True,
                    collate_fn=train_sampler.episodic_collate_fn,
            )

            auroc_mean, auroc_std, dauprc_mean, dauprc_std, bacc_mean, bacc_std, f1_mean, f1_std, kappa_mean, kappa_std = evaluate(
                    original_maml=maml, 
                    data_loader=test_loader, 
                    device=device, 
                    adaptation_steps=adaptation_steps, 
                    loss=loss
                )
            final_result_auroc[str(support_set_size)].append(f"{auroc_mean:.2f}+/-{auroc_std:.2f}")
            final_result_dauprc[str(support_set_size)].append(f"{dauprc_mean:.2f}+/-{dauprc_std:.2f}")
            final_result_bacc[str(support_set_size)].append(f"{bacc_mean:.2f}+/-{bacc_std:.2f}")
            final_result_f1[str(support_set_size)].append(f"{f1_mean:.2f}+/-{f1_std:.2f}")
            final_result_kappa[str(support_set_size)].append(f"{kappa_mean:.2f}+/-{kappa_std:.2f}")

    df_assay_id_map = pd.read_csv(df_assay_id_map_path)
    df_assay_id_map = df_assay_id_map.astype({'ASSAY_ID': str})

    df_score = pd.DataFrame(data=final_result_auroc)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path1, index=False)

    df_score = pd.DataFrame(data=final_result_dauprc)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path2, index=False)

    df_score = pd.DataFrame(data=final_result_bacc)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path3, index=False)

    df_score = pd.DataFrame(data=final_result_f1)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path4, index=False)

    df_score = pd.DataFrame(data=final_result_kappa)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path5, index=False)


if __name__ == '__main__':
    main()