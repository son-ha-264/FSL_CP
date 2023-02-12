import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


import torch
from torch.nn.modules import Linear
import torch.optim as optim

from utils.metrics import multitask_bce
from datamodule.multitask_cp_2 import prepare_support_query_multitask_cp, load_FNN_with_trained_weights
from torch.utils.data import DataLoader
from utils.models.shared_models import FNN_Relu


### Inits
num_repeat = 50
support_set_sizes = [8, 16, 32, 64, 96]
query_set_size = 32
max_epochs = 50
loss_function = multitask_bce()
sigmoid = torch.nn.Sigmoid()

### Paths inits
HOME = os.environ['HOME']
data_folder = os.path.join(HOME, 'FSL_CP/data/output')
df_assay_id_map_path = os.path.join(HOME, 'FSL_CP/data/output/assay_target_map.csv') 
result_summary_path = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_cp_result_summary3.csv') 

### Open feature df
feature_df = pd.read_csv(os.path.join(data_folder, 'norm_CP_feature_df.csv'))
feature_df = feature_df.dropna(axis=1, how='any')
feature_df = feature_df.drop(columns=['INCHIKEY', 'CPD_SMILES', 'SAMPLE_KEY'])

pretrain_logs_path = '/home/son.ha/FSL_CP/logs/multitask_only_cp_pretrain'
path_to_weight = os.path.join(pretrain_logs_path, 'lightning_logs/version_0/checkpoints/final_model.ckpt')
logs_path = '/home/son.ha/FSL_CP/logs/multitask_cp_finetune2'


### Final result dictionary
final_result = {
        '8': [],
        '16': [],
        '32': [],
        '64': [],
        '96':[]
}

# Load the assay key
with open(os.path.join(data_folder, 'data_split.json')) as f:
    data = json.load(f)
test_split = data['test']
final_result['ASSAY_ID'] = test_split

### Loop through all support set sizes
for support_set_size in tqdm(support_set_sizes, desc='Support set size'):
    tqdm.write(f"Analysing for support set size {support_set_size}")
    for test_assay in tqdm(test_split, desc='Test split', leave=False):
        list_auc = []
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
            support_loader = DataLoader(support_set, batch_size=256,
                                    shuffle=False, num_workers=20)
            
            query_loader = DataLoader(query_set, batch_size=256,
                                    shuffle=False, num_workers=20)

            
            ### Load pretrained models
            fnn_pretrained = load_FNN_with_trained_weights(
                path_to_weight=path_to_weight,
                input_shape=len(support_set[3][0]),
            )
            fnn_pretrained.classifier[10] = Linear(in_features=2048, out_features=1, bias=True)
            fnn_pretrained = fnn_pretrained.cuda()
            
            # Fine-tune
            optimizer = optim.SGD(fnn_pretrained.parameters(), lr=0.001, momentum=0.9)
            for epoch in range(max_epochs):
                    for i, (inputs, labels) in enumerate(support_loader, 0):
                        optimizer.zero_grad()
                        outputs = fnn_pretrained(inputs.cuda())
                        loss = loss_function(outputs, labels.cuda())
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(fnn_pretrained.parameters(), max_norm=5)
                        optimizer.step()

            # Inference
            fnn_pretrained.eval()
            list_pred = []
            list_true = []
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(query_loader, 0):
                    pred = fnn_pretrained(inputs.cuda())
                    pred = sigmoid(pred)
                    list_pred.append(pred)
                    list_true.append(labels)
            tensor_pred = torch.cat(list_pred, 1)
            pred_array = tensor_pred.cpu().detach().numpy()
            true_array = labels.detach().numpy()
            list_auc.append(roc_auc_score(true_array, pred_array))
        final_result[str(support_set_size)].append(f"{np.mean(list_auc):.2f}+/-{np.std(list_auc):.2f}")

### Create result summary dataframe
df_assay_id_map = pd.read_csv(df_assay_id_map_path)
df_assay_id_map = df_assay_id_map.astype({'ASSAY_ID': str})
df_score = pd.DataFrame(data=final_result)
df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
df_final.to_csv(result_summary_path, index=False)