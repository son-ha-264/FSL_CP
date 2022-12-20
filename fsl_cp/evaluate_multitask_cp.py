import numpy as np
import pandas as pd
import glob
import torch
import os

from sklearn.metrics import roc_auc_score


def AUC_with_missing_data(target, pred):
    """
    Calculate per-assay AUC for a batch of predictions and targets. 
    Each column denotes an assay
    Missing data is denoted as -1. Not include missing data when calculate AUC
    Input: 2 arrays with missing values
    Output AUC Score
    """ 
    pred = np.array(pred)
    target = np.array(target)
    available_data = np.where(target>=0)
    pred = pred[available_data]
    target = target[available_data]
    return roc_auc_score(target, pred)


def main():

    ### Inits
    support_set_sizes = [16, 32, 64, 96]
    num_repeat = 10
    path_to_file = '/home/son.ha/FSL_CP/output/multitask_pred_3'
    df_assay_id_map_path = "/home/son.ha/FSL_CP/data/output/assay_target_map.csv"
    result_summary_path = '/home/son.ha/FSL_CP/result/result_summary/multitask_cp+_result_summary_2.csv'

    final_result = {
        'ASSAY_ID': [],
        16:[],
        32:[],
        64:[],
        96:[]
    } 

    final_result['ASSAY_ID'] = list(pd.read_csv(os.path.join(path_to_file, f"query_set_16_0.csv"), index_col=0).columns[1:])

    for support_set_size in support_set_sizes:
        temp_auc = []
        for repeat in range(num_repeat):
            y_true = pd.read_csv(os.path.join(path_to_file, f"query_set_{support_set_size}_{repeat}.csv"), index_col=0)
            y_true = y_true.drop(columns='NUM_ROW_CP_FEATURES')
            y_pred = torch.load(os.path.join(path_to_file, f"multitask_pred_{support_set_size}_{repeat}.pt"))
            y_pred = torch.cat(y_pred)
            y_pred = pd.DataFrame(data=y_pred.numpy())
            temp_auc_this_repeat = []
            for i in range(len(final_result['ASSAY_ID'])):
                pred = y_pred.iloc[:,i]
                label = y_true.iloc[:,i]
                temp_auc_this_repeat.append(AUC_with_missing_data(label, pred))
            temp_auc.append(temp_auc_this_repeat)
        temp_auc = np.array(temp_auc)
        temp_auc_mean = np.mean(temp_auc, axis=0)
        temp_auc_std = np.std(temp_auc, axis=0)
        final_result[support_set_size]= [f"{temp_auc_mean[i]:.2f}+/-{temp_auc_std[i]:.2f}" for i in range(len(final_result['ASSAY_ID']))]
    
    df_assay_id_map = pd.read_csv(df_assay_id_map_path)
    df_assay_id_map = df_assay_id_map.astype({'ASSAY_ID': str})
    df_score = pd.DataFrame(data=final_result)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path, index=False)
    return None


if __name__ == '__main__':
    main()
