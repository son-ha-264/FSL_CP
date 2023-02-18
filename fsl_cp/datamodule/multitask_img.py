import random
import os
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


def _change_assay_to_columns(df):
        newdf = df.pivot(index='NUM_ROW_CP_FEATURES', columns='ASSAY', values="LABEL")
        newdf = newdf.rename(columns={newdf.columns[0]: 'LABEL'})
        return(newdf)

def prepare_support_query_multitask_cp(
    assay_code: str,
    label_df_path: str,
    feature_df: pd.core.frame.DataFrame,
    support_set_size: int,
    query_set_size: int,
    random_state=None,

):

    # Inits
    if not random_state:
        random_state = random.randint(0,100)

    # Load label csv file
    label_df_before = pd.read_csv(label_df_path)
    label_df_before['ASSAY'] = label_df_before['ASSAY'].astype(str)
    label_df_before = label_df_before[label_df_before['ASSAY'].isin([assay_code])]
    chosen_assay_df = label_df_before.reset_index(drop=True)

    # If random sample for few-shot prediction
    chosen_assay_df_2, support_set_df, _unused1, label_support = train_test_split(
                chosen_assay_df, chosen_assay_df['LABEL'], test_size=support_set_size, 
                stratify=chosen_assay_df['LABEL'], random_state=random_state
            )
    _unused_2, query_set_df, _unused3, label_query = train_test_split(
                            chosen_assay_df_2, chosen_assay_df_2['LABEL'], test_size=query_set_size, 
                            stratify=chosen_assay_df_2['LABEL'], random_state=random_state
                        )

    support_set_df = _change_assay_to_columns(support_set_df).dropna(axis=0, how='all').reset_index(drop=False)
    query_set_df = _change_assay_to_columns(query_set_df).dropna(axis=0, how='all').reset_index(drop=False)
    support_set_df = support_set_df.fillna(-1)
    query_set_df = query_set_df.fillna(-1)

    return multitask_cp_dataset(feature_df, support_set_df), multitask_cp_dataset(feature_df, query_set_df)


class multitask_cp_dataset(Dataset):
    def __init__(self,
            feature_df: pd.core.frame.DataFrame,
            label_df: pd.core.frame.DataFrame,):
        super().__init__()
        self.label_df = label_df
        self.feature_df = feature_df
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        cp_f_df_idx = self.label_df['NUM_ROW_CP_FEATURES'][idx]
        x = torch.tensor(self.feature_df.iloc[cp_f_df_idx, :],dtype=torch.float)
        y = torch.tensor(self.label_df.loc[idx, 'LABEL'],dtype=torch.float)
        return x,y 
    
    def return_df(self):
        return self.feature_df, self.label_df


class multitask_pretrain_img_dataset(Dataset):
    """Pytorch dataset class for multitask pretraining with CP profile
    
    Args:
        assay_codes: List of assay codes
        cp_f_paths: 
    """
    def __init__(self,
                assay_codes: List[str],
                image_path:str,
                label_df: pd.core.frame.DataFrame,
                transform=None
                                
    ):
        super(multitask_pretrain_img_dataset).__init__()

        # Load label csv file
        self.label_df_before = label_df
        self.label_df_before['ASSAY'] = self.label_df_before['ASSAY'].astype(str)
        self.label_df_before = self.label_df_before[self.label_df_before['ASSAY'].isin(assay_codes)]
        self.label_df_before = self.label_df_before.reset_index(drop=True)
        self.label_df = self.label_df_before[['LABEL', 'ASSAY', 'SAMPLE_KEY_VIEW']].pivot_table(index='SAMPLE_KEY_VIEW', columns='ASSAY')
        self.label_df.columns = self.label_df.columns.droplevel(0)
        self.label_df = self.label_df.reset_index(drop=False)
        self.label_df = self.label_df.fillna(-1)

        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        filename = self.label_df.loc[idx, 'SAMPLE_KEY_VIEW'] + '.npz'
        filename = os.path.join(self.image_path, filename)
        x = np.load(filename)["sample"]
        y = torch.tensor(self.label_df.iloc[idx, 1:],dtype=torch.float)
        if self.transform:
            x = self.transform(x)
        return x,y 


def load_FNN_with_trained_weights(path_to_weight: str, input_shape, map_location=torch.device('cuda:0')):
    """Return FNN with trained weights 
       Change which GPU/CPU to load the model on with map_loaction 
    """
    checkpoint = torch.load(path_to_weight, map_location=map_location)
    fnn = FNN_Relu(num_classes=checkpoint['hyper_parameters']['num_classes'], input_shape=input_shape)
    weights = checkpoint["state_dict"]
    del weights['sigma']
    keys = list(weights.keys())
    values = weights.values()
    new_keys = [i[6:] for i in keys]
    new_dict = collections.OrderedDict(zip(new_keys, values))
    fnn.load_state_dict(new_dict)
    return fnn


def each_view_a_datapoint(df):
    '''Helper function. Make eachrow of the dataframe a view, instead of a well.
    Used for multitask pretraining.'''
    df['VIEWS_LIST'] = df['VIEWS'].apply(lambda s : s.split('_'))
    df = df.explode('VIEWS_LIST', ignore_index=True)
    df['SAMPLE_KEY_VIEW'] = df[['SAMPLE_KEY', 'VIEWS_LIST']].agg('-'.join, axis=1)
    df = df.drop(columns=['VIEWS_LIST'])
    return df