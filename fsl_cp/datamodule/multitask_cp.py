import random
import torch
from typing import Dict, List
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn 
import collections
from sklearn.model_selection import train_test_split
from utils.models.shared_models import FNN_Relu
from pandas.errors import DtypeWarning

# Ignore unecessary warning
import warnings
warnings.simplefilter(action='ignore', category=DtypeWarning)

class multitask_pretrain_cp_dataset(Dataset):
    """Pytorch dataset class for multitask pretraining with CP profile
    
    Args:
        assay_codes: List of assay codes
        cp_f_paths: 
    """
    def __init__(self,
                assay_codes: List[str],
                label_df_path: str,
                cp_f_path: List[str],
                inference: bool,
                set_size=32,
                random_state=None,
                support=True
                                
    ):
        super(multitask_pretrain_cp_dataset).__init__()

        # Inits
        if random_state:
            self.random_state = random_state
        else:
            self.random_state = random.randint(0,100)
        self.set_size = set_size

        # Load label csv file
        self.label_df_before = pd.read_csv(label_df_path)
        self.label_df_before['ASSAY'] = self.label_df_before['ASSAY'].astype(str)
        self.label_df_before = self.label_df_before[self.label_df_before['ASSAY'].isin(assay_codes)]
        self.label_df_before = self.label_df_before.reset_index(drop=True)

        # If random sample for few-shot prediction
        # NOTE: Have to do this roundabout to sample seperate support and query sets. 
        if inference:
            list_df = []
            for sampled_task in assay_codes:
                chosen_assay_df = self.label_df_before[self.label_df_before['ASSAY'] == sampled_task]
                chosen_assay_df_2, support_set_df, _unused1, label_support = train_test_split(
                            chosen_assay_df, chosen_assay_df['LABEL'], test_size=self.set_size, 
                            stratify=chosen_assay_df['LABEL'], random_state=self.random_state
                        )
                _unused_2, query_set_df, _unused3, label_query = train_test_split(
                                        chosen_assay_df_2, chosen_assay_df_2['LABEL'], test_size=set_size, 
                                        stratify=chosen_assay_df_2['LABEL'], random_state=self.random_state
                                    )
                if support:
                    list_df.append(support_set_df[['NUM_ROW_CP_FEATURES', 'LABEL', 'ASSAY']])
                else: 
                    list_df.append(query_set_df[['NUM_ROW_CP_FEATURES', 'LABEL', 'ASSAY']])
            list_df_modified = [self._change_assay_to_columns(f) for f in list_df]
            self.label_df = pd.concat(list_df_modified, axis=1).dropna(axis=0, how='all').reset_index(drop=False)
            self.label_df = self.label_df.fillna(-1)

        # Else, if pretrain
        else:
            self.label_df = self.label_df_before[['LABEL', 'ASSAY', 'NUM_ROW_CP_FEATURES']].pivot_table(index='NUM_ROW_CP_FEATURES', columns='ASSAY')
            self.label_df.columns = self.label_df.columns.droplevel(0)
            self.label_df = self.label_df.reset_index(drop=False)
            self.label_df = self.label_df.fillna(-1)

        # Read feature matrices and concat them
        list_feature_df = []
        for path in cp_f_path:
            feature_df = pd.read_csv(path)
            feature_df = feature_df.dropna(axis=1, how='any')
            feature_df = feature_df.drop(columns=['INCHIKEY', 'CPD_SMILES', 'SAMPLE_KEY'])
            list_feature_df.append(feature_df)
        self.feature_df = pd.concat(list_feature_df, axis=1)

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        cp_f_df_idx = self.label_df['NUM_ROW_CP_FEATURES'][idx]
        x = torch.tensor(self.feature_df.iloc[cp_f_df_idx, :],dtype=torch.float)
        y = torch.tensor(self.label_df.iloc[idx, 1:],dtype=torch.float)
        return x,y 

    def _change_assay_to_columns(self, df):
        newdf = df.pivot_table(index='NUM_ROW_CP_FEATURES', columns='ASSAY')
        newdf.columns = newdf.columns.droplevel(0)
        return(newdf)

    def get_label_df(self):
        return(self.label_df)

'''
class FNN_Relu(nn.Module):
    """
    Fully-connected neural network with ReLU activations.
    Model credit: https://github.com/ml-jku/hti-cnn 
    """
    def __init__(self, model_params=None, num_classes=1600, input_shape=None):
        super(FNN_Relu, self).__init__()
        assert input_shape
        fc_units = 2048
        drop_prob = 0.5
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_shape, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, num_classes),
        )
        
        # init
        self.init_parameters()
    
    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)
'''

def load_FNN_with_trained_weights(path_to_weight: str, input_shape):
    """Return FNN with trained weights 
    """
    checkpoint = torch.load(path_to_weight)
    fnn = FNN_Relu(num_classes=checkpoint['hyper_parameters']['num_classes'], input_shape=input_shape)
    weights = checkpoint["state_dict"]
    del weights['sigma']
    keys = list(weights.keys())
    values = weights.values()
    new_keys = [i[6:] for i in keys]
    new_dict = collections.OrderedDict(zip(new_keys, values))
    fnn.load_state_dict(new_dict)
    return fnn


"""
class multitask_inference_cp_dataset(Dataset):
    '''Pytorch dataset class for multitask pretraining with CP profile
    Args:  
        assay_codes: List of codes for assays
        label_df_path: path to label df
        cp_f_path: List of paths to features dfs
        query_set_size: Size of the query set
    '''
    def __init__(self,
                assay_codes: List[str],
                label_df_path: str,
                cp_f_path: List[str],
                query_set_size: int,

    ):
        super(multitask_inference_cp_dataset).__init__()

        # Load label csv file
        self.label_df_before = pd.read_csv(label_df_path)
        self.label_df_before['ASSAY'] = self.label_df_before['ASSAY'].astype(str)
        self.label_df_before = self.label_df_before[self.label_df_before['ASSAY'].isin(assay_codes)]
        self.label_df_before = self.label_df_before.reset_index(drop=True)

        # Random sample for few-shot prediction
        list_df = []
        for sampled_task in assay_codes:
            chosen_assay_df = self.label_df_before[self.label_df_before['ASSAY'] == sampled_task]
            _unused_2, query_set_df, _unused3, label_query = train_test_split(
                                    chosen_assay_df, chosen_assay_df['LABEL'], test_size=query_set_size, stratify=chosen_assay_df['LABEL']
                                )
            list_df.append(query_set_df[['NUM_ROW_CP_FEATURES', 'LABEL', 'ASSAY']])
        list_df_modified = [self._change_assay_to_columns(f) for f in list_df]
        self.label_df = pd.concat(list_df_modified, axis=1).dropna(axis=0, how='all').reset_index(drop=False)
        self.label_df = self.label_df.fillna(-1)

        # Read feature matrices and concat them
        list_feature_df = []
        for path in cp_f_path:
            feature_df = pd.read_csv(path)
            feature_df = feature_df.dropna(axis=1, how='any')
            feature_df = feature_df.drop(columns=['INCHIKEY', 'CPD_SMILES', 'SAMPLE_KEY'])
            list_feature_df.append(feature_df)
        self.feature_df = pd.concat(list_feature_df, axis=1)

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        cp_f_df_idx = self.label_df['NUM_ROW_CP_FEATURES'][idx]
        x = torch.tensor(self.feature_df.iloc[cp_f_df_idx, :],dtype=torch.float)
        y = torch.tensor(self.label_df.iloc[idx, 1:],dtype=torch.float)
        return x,y 

    def _change_assay_to_columns(self, df):
        newdf = df.pivot_table(index='NUM_ROW_CP_FEATURES', columns='ASSAY')
        newdf.columns = newdf.columns.droplevel(0)
        return(newdf)
"""
""" class Finetuning_CP_Multitask_Data(Dataset):
    '''
    Input: csv features, list num_row_feat, list label
    '''
    def __init__(self, num_row_f, label, csv_f_path="csv/CP_features.csv"):
        super().__init__()
        self.df_f = pd.read_csv(csv_f_path)
        self.num_row_f = num_row_f
        self.label = label
        assert len(self.label) == len(self.num_row_f)

        self.df_f = self.df_f.drop(columns='INCHIKEY')
        self.df_f = (self.df_f - self.df_f.mean())/self.df_f.std()
        self.df_f = self.df_f.dropna(axis=1)
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        num_row = self.num_row_f[idx]
        x = torch.tensor(self.df_f.iloc[num_row, :],dtype=torch.float)
        y = torch.tensor(self.label[idx],dtype=torch.long)
        return x,y

def sliding_average(value_list: List[float], window: int) -> float:
    '''
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats (can't be empty)
        window: number of instances to take into account. If value is 0 or greater than
            the length of value_list, all instances will be taken into account.
    Returns:
        average of the last window instances in value_list
    Raises:
        ValueError: if the input list is empty
    '''
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()

def single_task_sampler(
        assay_code,
        n_shot,
        n_query, 
        query_sample="random",
        num_repeat=64 
    ):
    '''
        Given an assay code, balance sample the support set and random sample the query set 
    Returns an iterator that returns the sampled sets
    '''
    assert query_sample in ["balance", "random"]

    jsonl_path = 'data_short/'+assay_code+'.jsonl'
    d_actives = []
    d_inactives = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            if obj['LABEL'] == 1:
                d_actives.append((obj['NUM_ROW_CP_FEATURES'], obj['LABEL']))
            elif obj['LABEL'] == 0:
                d_inactives.append((obj['NUM_ROW_CP_FEATURES'], obj['LABEL']))

    assert len(d_actives)>n_shot+n_query and len(d_inactives)>n_shot+n_query

    for _ in range(num_repeat):
        temp_actives = d_actives
        temp_inactives = d_inactives
        random.shuffle(temp_actives)
        random.shuffle(temp_inactives)
        support_set = temp_actives[0:n_shot] + temp_inactives[0:n_shot]
        if query_sample == "random":
            query_set = random.sample(temp_actives[n_shot:]+temp_inactives[n_shot:], n_query*2)
        elif query_sample == "balance":
            query_set = temp_actives[n_shot:n_shot+n_query] + temp_inactives[n_shot:+n_query]
        support_num_row_f = [i[0] for i in support_set]
        support_label = [i[1] for i in support_set]
        query_num_row_f = [i[0] for i in query_set]
        query_label = [i[1] for i in query_set]
        yield support_num_row_f, support_label, query_num_row_f, query_label """