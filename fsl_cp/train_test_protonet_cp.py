import torch 
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
import json, codecs
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.models.protonet import ProtoNet, FNN_Relu
from datamodule.protonet_cp import protonet_cp_dataset, protonet_cp_sampler
import os
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from utils.metrics import delta_auprc


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
    """ Fit function for pretraining protonet. 
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
    #if save_path:
    #    np.save(save_path, np.array(scores))
    return np.mean(AUROC_scores), np.std(AUROC_scores), np.mean(dAUPRC_scores), np.std(dAUPRC_scores)

def main(
        seed=69
):
    ### Seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    ### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--device', type=str, default='cuda:0',
        help='cuda, cuda:0 or cpu')
    args = parser.parse_args()

    device = args.device
    
    ### Inits
    if device == 'cpu':
        device = torch.device('cpu')
    elif 'cuda' in device and torch.cuda.device_count():
        device = torch.device(device)

    ### Inits
    support_set_sizes = [8, 16, 32, 64, 96]
    query_set_size = 32
    num_episodes_train = 70000
    num_episodes_test = 100
    step_size = 20000

    HOME = os.environ['HOME']

    json_path = os.path.join(HOME, 'FSL_CP/data/output/data_split.json')
    label_df_path= os.path.join(HOME, 'FSL_CP/data/output/FINAL_LABEL_DF.csv')
    cp_f_path=[os.path.join(HOME,'FSL_CP/data/output/norm_CP_feature_df.csv')]
    df_assay_id_map_path = os.path.join(HOME, 'FSL_CP/data/output/assay_target_map.csv') 
    result_summary_path1 = os.path.join(HOME, 'FSL_CP/result/result_summary/protonet_cp_auroc_result_summary.csv') 
    result_summary_path2 = os.path.join(HOME, 'FSL_CP/result/result_summary/protonet_cp_dauprc_result_summary.csv') 

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
    for support_set_size in support_set_sizes:
        tqdm.write(f"Analysing for support set size {support_set_size}")

        # Load train data
        train_data = protonet_cp_dataset(
            train_split, 
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

        # Load model
        input_shape=len(train_data[3][0])
        backbone = FNN_Relu(num_classes=512, input_shape=input_shape)
        model = ProtoNet(backbone).to(device)

        # Pretrain on random assays
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
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

        '''
        # Performance before pretraining
        before_mean, before_std = evaluate(
            model, 
            test_loader, 
            #save_path=os.path.join(temp_folder, f"protonet_before_{support_set_size}_{test_assay}.npy")
        )
        result_before_pretrain[str(support_set_size)+'_auc_before_train'].append(f"{before_mean:.2f}+/-{before_std:.2f}")
        '''

        for test_assay in tqdm(test_split):
            
            # Load test data
            test_data = protonet_cp_dataset(
                test_split, 
                label_df_path= label_df_path, 
                cp_f_path=cp_f_path
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

            # Performance after pretraining
            auroc_mean, auroc_std, dauprc_mean, dauprc_std = evaluate(
                model, 
                test_loader, 
                device
                #save_path=os.path.join(temp_folder, f"protonet_after_{support_set_size}_{test_assay}.npy")
            )
            final_result_auroc[str(support_set_size)].append(f"{auroc_mean:.2f}+/-{auroc_std:.2f}")
            final_result_dauprc[str(support_set_size)].append(f"{dauprc_mean:.2f}+/-{dauprc_std:.2f}")

    # Create result summary dataframe
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