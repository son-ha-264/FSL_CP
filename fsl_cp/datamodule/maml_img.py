import os
import random
import warnings
from typing import List, Tuple
import pandas as pd
import numpy as np
from pandas.errors import DtypeWarning

import torch
from torch import Tensor
from torch.utils.data import Sampler
from torch.utils.data import Dataset

from .utils import task_sample

warnings.filterwarnings("ignore", category=DtypeWarning)

class maml_img_dataset(Dataset):
    """DataLoader for ProtoNet running on CP profiles. 
       Input JSONL files, convert them to lists of inputs and labels

    Args:
        assay_code: list of code numbers of assay
        data_folder: path to jsonl data files
        cp_f_path: list of paths to csv files with features (e.g. ecfp,...)
    """

    def __init__(self, 
                assay_codes: List[str], 
                label_df_path: str,
                image_path: str,
                transform=None
    ):
        super().__init__()

        # Miscellanous inits
        self.image_path = image_path
        self.transform = transform

        # Load label csv file
        self.label_df = pd.read_csv(label_df_path)
        self.label_df['ASSAY'] = self.label_df['ASSAY'].astype(str)
        self.label_df = self.label_df[self.label_df['ASSAY'].isin(assay_codes)]
        self.label_df = self.label_df.reset_index(drop=True)
        self.label_df['SAMPLE_KEY_VIEW'] = self.label_df[['SAMPLE_KEY', 'VIEWS']].agg('-'.join, axis=1).str[0:11]

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        filename = self.label_df.loc[idx, 'SAMPLE_KEY_VIEW'] + '.npz'
        filename = os.path.join(self.image_path, filename)
        x = np.load(filename)["sample"]
        y = self.label_df.loc[idx, 'LABEL']
        if self.transform:
            x = self.transform(x)
        return x,y 

    def get_df(self):
        return self.label_df
    


class maml_img_sampler(Sampler):
    """
    Custom sampler for few-shot prediction using ProtoNet
    When construct a Pytorch DataLoader, set the collate_fn argument as: 

        collate_fn=train_sampler.episodic_collate_fn

    (refer to the script 'train_test_protonet_img.py' for exact usage)

    Args:
        task_dataset: A protonet_img_dataset data object
        support_set_size: size of support set (=2 * num_shots)
        query_set_size: size of query set
        num_episodes: Number of training episodes
        specific_assay: Code of one assay you want to sample (mainly for debugging)
    """
    def __init__(
        self,
        task_dataset: maml_img_dataset,
        support_set_size: int,
        query_set_size: int,
        meta_batch_size:int, 
        num_episodes: int,
        sample_method = 'stratify',
        specific_assay = None,
    ):
        super().__init__(data_source=None)

        # Misc inits
        self.support_set_size = support_set_size
        self.query_set_size = query_set_size
        self.num_episodes = num_episodes * meta_batch_size
        self.task_dataset = task_dataset
        self.df = self.task_dataset.get_df()
        self.specific_assay = specific_assay
        self.sample_method = sample_method

        # Extract list of assays
        self.assay_list = list(self.df['ASSAY'].unique())

    def __len__(self):
        return self.num_episodes
        
    def __iter__(self):
        for _ in range(self.num_episodes):
            
            bad_task = True
            # If anything wrong with the sampling process (due to data), just samples again
            while bad_task:
                try:
                    # Randomly choose an assay to sample from
                    if self.specific_assay:
                        sampled_task = self.specific_assay
                    else:
                        sampled_task = random.sample(self.assay_list, 1)[0]

                    # Sample supoprt and query sets from the chosen assay 
                    chosen_assay_df = self.df[self.df['ASSAY'] == sampled_task]
                    support_set_df, query_set_df, label_support, label_query = task_sample(self.sample_method, chosen_assay_df, self.support_set_size, self.query_set_size)

                    # Sample sanity check
                    assert len(support_set_df) == self.support_set_size
                    assert len(query_set_df) == self.query_set_size
                    assert len(set(label_support)) == 2
                    assert len(set(label_query)) == 2

                    list_data_idx = list(support_set_df.index) + list(query_set_df.index) 

                except:
                    if self.specific_assay:
                        raise ValueError('Choose another assay that has more datapoints')
                    pass
                else: 
                    bad_task = False
            
            # Yield a list, each element correspond to the index of a row in the df
            yield list_data_idx

    def episodic_collate_fn(
        self,
        input_data: List[Tuple[Tensor, int]]
    ):
        true_class_ids = list({x[1] for x in input_data})
    
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.support_set_size+self.query_set_size, *all_images.shape[1:])
        )

        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.support_set_size+self.query_set_size))

        support_images = all_images[: self.support_set_size].reshape(
            (-1, *all_images.shape[1:])
        )

        query_images = all_images[self.support_set_size :].reshape((-1, *all_images.shape[1:]))
        support_labels = all_labels[: self.support_set_size].flatten()
        query_labels = all_labels[self.support_set_size :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )