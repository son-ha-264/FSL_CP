import os
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch.nn.modules import Linear
import torch.optim as optim
import torch.nn as nn

from utils.metrics import multitask_bce
from datamodule.multitask_cp import prepare_support_query_multitask_cp, load_FNN_with_trained_weights
from torch.utils.data import DataLoader
from utils.models.shared_models import FNN_Relu
from utils.metrics import delta_auprc
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score


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
        help='cuda:0 (or whichever gpu) or cpu.')
    parser.add_argument(
        '-c', '--checkpoint_path', type=str, default=None,
        help='path to ckpt file containing model weights.')
    args = parser.parse_args()

    path_to_weight = args.checkpoint_path
    device = args.device
    if device == 'cpu':
        device = torch.device('cpu')
    elif 'cuda' in device and torch.cuda.device_count():
        device = torch.device(device)

    ### Inits
    num_repeat = 50
    support_set_sizes = [8, 16, 32, 64, 96]
    query_set_size = 32
    max_epochs = 10
    loss_function = nn.BCEWithLogitsLoss()
    sigmoid = torch.nn.Sigmoid()

    ### Paths inits
    HOME = os.environ['HOME']
    data_folder = os.path.join(HOME, 'FSL_CP/data/output')
    df_assay_id_map_path = os.path.join(HOME, 'FSL_CP/data/output/assay_target_map.csv') 
    result_summary_path1 = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_cp_auroc_result_summary.csv') 
    result_summary_path2 = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_cp_dauprc_result_summary.csv') 
    result_summary_path3 = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_cp_bacc_result_summary.csv') 
    result_summary_path4 = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_cp_f1_result_summary.csv') 
    result_summary_path5 = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_cp_kappa_result_summary.csv') 

    ### Open feature df
    feature_df = pd.read_csv(os.path.join(data_folder, 'norm_CP_feature_df.csv'))
    feature_df = feature_df.dropna(axis=1, how='any')
    feature_df = feature_df.drop(columns=['INCHIKEY', 'CPD_SMILES', 'SAMPLE_KEY'])

    if not path_to_weight:
        path_to_weight = os.path.join(HOME, 'FSL_CP/weights/multitask_cp/final_model.ckpt')

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

    # Load the assay key
    with open(os.path.join(data_folder, 'data_split.json')) as f:
        data = json.load(f)
    test_split = data['test']
    final_result_auroc['ASSAY_ID'] = test_split
    final_result_dauprc['ASSAY_ID'] = test_split
    final_result_bacc['ASSAY_ID'] = test_split
    final_result_f1['ASSAY_ID'] = test_split
    final_result_kappa['ASSAY_ID'] = test_split

    ### Loop through all support set sizes
    for support_set_size in tqdm(support_set_sizes, desc='Support set size'):
        tqdm.write(f"Analysing for support set size {support_set_size}")
        for test_assay in tqdm(test_split, desc='Test split', leave=False):
            list_auroc = []
            list_dauprc = []
            list_bacc = []
            list_f1 = []
            list_kappa = []
            for rep in tqdm(range(num_repeat), desc='Num repeat', leave=False):
                ### Load data (support and query sets)
                support_set, query_set = prepare_support_query_multitask_cp(
                    assay_code=test_assay,
                    label_df_path=os.path.join(data_folder, 'FINAL_LABEL_DF.csv'),
                    feature_df=feature_df,
                    support_set_size=support_set_size,
                    query_set_size=query_set_size,
                )


                ### Create DataLoader objects
                support_loader = DataLoader(support_set, batch_size=int(support_set_size/4),
                                        shuffle=True, num_workers=20)
                
                query_loader = DataLoader(query_set, batch_size=query_set_size,
                                        shuffle=False, num_workers=20)

                
                ### Load pretrained models
                #fnn_pretrained = load_FNN_with_trained_weights(
                #    path_to_weight=path_to_weight,
                #    input_shape=len(support_set[3][0]),
                #)
                #fnn_pretrained.classifier[10] = Linear(in_features=2048, out_features=1, bias=True)
                #fnn_pretrained = fnn_pretrained.to(device)

                fnn_pretrained = FNN_Relu(input_shape=len(support_set[3][0]), num_classes=1).to(device)

                # Fine-tune
                #optimizer = optim.SGD(fnn_pretrained.parameters(), lr=0.001, momentum=0.9)
                optimizer = optim.Adam(fnn_pretrained.parameters(), 0.001)
                for epoch in range(max_epochs):
                        for i, (inputs, labels) in enumerate(support_loader, 0):
                            optimizer.zero_grad()
                            outputs = fnn_pretrained(inputs.to(device))
                            loss = loss_function(torch.squeeze(outputs), labels.to(device))
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(fnn_pretrained.parameters(), max_norm=5)
                            optimizer.step()

                # Inference
                fnn_pretrained.eval()
                list_pred = []
                list_true = []
                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(query_loader, 0):
                        pred = fnn_pretrained(inputs.to(device))
                        pred = sigmoid(pred)
                        list_pred.append(pred)
                        list_true.append(labels)
                tensor_pred = torch.cat(list_pred, 1)
                pred_array = tensor_pred.cpu().detach().numpy()
                true_array = labels.detach().numpy()
                list_auroc.append(roc_auc_score(true_array, pred_array))
                list_dauprc.append(delta_auprc(true_array, pred_array))                
                list_bacc.append(balanced_accuracy_score(true_array, np.rint(pred_array), adjusted=True))
                list_f1.append(f1_score(true_array, np.rint(pred_array)))
                list_kappa.append(cohen_kappa_score(true_array, np.rint(pred_array)))
                
            final_result_auroc[str(support_set_size)].append(f"{np.mean(list_auroc):.2f}+/-{np.std(list_auroc):.2f}")
            final_result_dauprc[str(support_set_size)].append(f"{np.mean(list_dauprc):.2f}+/-{np.std(list_dauprc):.2f}")
            final_result_bacc[str(support_set_size)].append(f"{np.mean(list_bacc):.2f}+/-{np.std(list_bacc):.2f}")
            final_result_f1[str(support_set_size)].append(f"{np.mean(list_f1):.2f}+/-{np.std(list_f1):.2f}")
            final_result_kappa[str(support_set_size)].append(f"{np.mean(list_kappa):.2f}+/-{np.std(list_kappa):.2f}")

     # Create result summary dataframe
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

    return None

if __name__ == '__main__':
    main()