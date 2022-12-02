from torch.utils.data import Dataset
from typing import List, Tuple
import pandas as pd
import torch
import os
import jsonlines
import random
from torch import Tensor
from torch.utils.data import Sampler
from sklearn.model_selection import train_test_split
import warnings
from pandas.errors import DtypeWarning
import numpy as np
import itertools

warnings.filterwarnings("ignore", category=DtypeWarning)

'''
Simulate 6 views just by sampling 6 times more datapoints
Need to create a custom sampler so that it samples in a stratified wasy, while sample all views from a compound 
'''

class protonet_img_dataset(Dataset):
    """DataLoader for ProtoNet running on CP profiles. 
       Input JSONL files, convert them to lists of inputs and labels

    Args:
        assay_code: list of code numbers of assay
        data_folder: path to jsonl data files
        cp_f_path: list of paths to csv files with features (e.g. ecfp,...)
        mode: 'pretrain'(mode 1) or 'inference'(mode 2) mode
    """

    def __init__(self, 
                assay_codes: List[str], 
                label_df_path: str,
                image_path: str,
                transform=None
    ):
        super(protonet_img_dataset).__init__()

        # Random inits
        self.image_path = image_path
        self.transform = transform

        # Load label csv file
        self.label_df = pd.read_csv(label_df_path)
        self.label_df['ASSAY'] = self.label_df['ASSAY'].astype(str)
        self.label_df = self.label_df[self.label_df['ASSAY'].isin(assay_codes)]
        self.label_df['NUM_VIEWS'] = self.label_df['VIEWS'].apply(lambda x: len(x.split('_')))
        self.label_df = self.label_df[self.label_df['NUM_VIEWS']==6]
        self.label_df = self.label_df.reset_index(drop=True)
        self.label_df['INDEX'] = range(len(self.label_df))

        # Untangled label csv (each view is a datapoint)
        a = self.label_df.apply(lambda x: self._create_img_id(x['SAMPLE_KEY'], x['VIEWS'], x['INDEX'], x['ASSAY'], x['LABEL']), axis=1)
        result = list(itertools.chain.from_iterable(a))
        self.untangled_df = pd.DataFrame(data={
            'df1_index':[int(i.split('/')[0]) for i in result],
            'SAMPLE_KEY_VIEW': [i.split('/')[1] for i in result],
            'ASSAY': [i.split('/')[2] for i in result],
            'LABEL': [np.float64(i.split('/')[3]) for i in result],
            })
        

    def __len__(self):
        return len(self.untangled_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        filename = self.untangled_df.loc[idx, 'SAMPLE_KEY_VIEW'] + '.npz'
        filename = os.path.join(self.image_path, filename)
        x = np.load(filename)["sample"]
        y = self.untangled_df.loc[idx, 'LABEL']
        if self.transform:
            x = self.transform(x)
        return x,y 

    def get_untangled_df(self):
        return self.untangled_df

    def get_df(self):
        return self.label_df

    def _create_img_id(self, sample_key, views, index, assay, label):
        return([str(index)+'/'+sample_key+'-'+i+'/'+str(assay)+'/'+str(label) for i in views.split('_')])


### TODO: strtified task sampler, random task sampler
### https://github.com/microsoft/FS-Mol/blob/main/fs_mol/data/fsmol_task_sampler.py


class protonet_img_sampler(Sampler):
    """
    1. Sample n tasks/assays from a total of N tasks
    2. For each task,
        a. Sample support and query compounds (wrt to compounds, not views)
        b. Return CP features and labels
    3. Return an iterator that yield () at each iteration

    Sampler (batch_sampler arg in DataLoader) yields a list of keys at a time
    """
    def __init__(
        self,
        task_dataset: protonet_img_dataset,
        support_set_size: int,
        query_set_size: int,
        num_episodes: int,
        specific_assay = None,
    ):
        super().__init__(data_source=None)
        #self.n_shots_support = int(support_set_size/2) #NOTE!: here is actually n-shots!!!, support_set = 2 x n_shots 
        #self.n_shots_query = int(query_set_size/2)

        # Inits
        self.support_set_size = support_set_size
        self.query_set_size = query_set_size
        self.img_support_set_size = support_set_size*6
        self.img_query_set_size = query_set_size*6
        self.num_episodes = num_episodes
        self.task_dataset = task_dataset
        self.label_df = task_dataset.get_df()
        self.untangled_df = task_dataset.get_untangled_df()
        self.specific_assay = specific_assay

        # Extract list of assays
        self.assay_list = list(self.untangled_df['ASSAY'].unique())

    def __len__(self):
        return self.num_episodes
        
    def __iter__(self):
        for _ in range(self.num_episodes):
            
            bad_task = True
            while bad_task:
                try:
                    # Randomly choose an assay to sample from
                    if self.specific_assay:
                        sampled_task = self.specific_assay
                    else:
                        sampled_task = random.sample(self.assay_list, 1)[0]

                    # Sample from the chosen assay 
                    chosen_assay_df = self.label_df[self.label_df['ASSAY'] == sampled_task]
                    chosen_assay_df_2, support_set_df, _unused1, label_support = train_test_split(
                        chosen_assay_df, chosen_assay_df['LABEL'], test_size=self.support_set_size, stratify=chosen_assay_df['LABEL']
                    )
                    _unused_2, query_set_df, _unused3, label_query = train_test_split(
                        chosen_assay_df_2, chosen_assay_df_2['LABEL'], test_size=self.query_set_size, stratify=chosen_assay_df_2['LABEL']
                    )
                    assert len(support_set_df) == self.support_set_size
                    assert len(query_set_df) == self.query_set_size
                    assert len(set(label_support)) == 2
                    assert len(set(label_query)) == 2

                    img_support_set_df = self.untangled_df[self.untangled_df['df1_index'].isin(list(support_set_df['INDEX']))].sort_values(by='df1_index')
                    img_query_set_df = self.untangled_df[self.untangled_df['df1_index'].isin(list(query_set_df['INDEX']))].sort_values(by='df1_index')
                    list_data_idx = list(img_support_set_df.index) + list(img_query_set_df.index) 

                    assert len(list_data_idx) == self.img_support_set_size + self.img_query_set_size
                except:
                    if self.specific_assay:
                        raise ValueError('Something wrong with the sampler')
                else: 
                    bad_task = False
            
            # Yield a list, each element correspond to a row in the df
            #for i in list(img_query_set_df.index):
            #    list_data_idx = list(img_support_set_df.index) + [i]
            yield list_data_idx

    def episodic_collate_fn(
        self,
        input_data: List[Tuple[Tensor, int]]
    ):
        true_class_ids = list({x[1] for x in input_data})
    
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.img_support_set_size+self.img_query_set_size, *all_images.shape[1:])
        )

        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.img_support_set_size+self.img_query_set_size))

        support_images = all_images[: self.img_support_set_size].reshape(
            (-1, *all_images.shape[1:])
        )

        query_images = all_images[self.img_support_set_size :].reshape((-1, *all_images.shape[1:]))
        support_labels = all_labels[: self.img_support_set_size].flatten()
        query_labels = all_labels[self.img_support_set_size :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )