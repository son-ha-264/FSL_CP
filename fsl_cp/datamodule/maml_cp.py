from torch.utils.data import Dataset
from typing import List, Tuple
import pandas as pd
import torch
import random
from torch import Tensor
from torch.utils.data import Sampler
from sklearn.model_selection import train_test_split
import warnings
from pandas.errors import DtypeWarning
from .utils import task_sample

warnings.filterwarnings("ignore", category=DtypeWarning)

class maml_cp_dataset(Dataset):
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
                cp_f_path: List[str],
    ):
        super().__init__()

        # Load label csv file
        self.label_df = pd.read_csv(label_df_path)
        self.label_df['ASSAY'] = self.label_df['ASSAY'].astype(str)
        self.label_df = self.label_df[self.label_df['ASSAY'].isin(assay_codes)]
        self.label_df = self.label_df.reset_index(drop=True)

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
        y = self.label_df['LABEL'][idx]
        return x,y 

    def get_label_df(self):
        return self.label_df

'''
def _sample_with_seed(seq, n, seed):
    if seed>=0:
        random.seed(seed)
    ans = random.sample(seq, n)
    return ans
'''


### TODO: strtified task sampler, random task sampler
### https://github.com/microsoft/FS-Mol/blob/main/fs_mol/data/fsmol_task_sampler.py


class maml_cp_sampler(Sampler):
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
        task_dataset: maml_cp_dataset,
        support_set_size: int,
        query_set_size: int,
        meta_batch_size:int, 
        num_episodes: int,
        specific_assay = None,
        sample_method = 'stratify'
    ):
        super().__init__(data_source=None)
        #self.n_shots_support = int(support_set_size/2) #NOTE!: here is actually n-shots!!!, support_set = 2 x n_shots 
        #self.n_shots_query = int(query_set_size/2)

        # Inits
        self.support_set_size = support_set_size
        self.query_set_size = query_set_size
        self.num_episodes = num_episodes * meta_batch_size
        self.task_dataset = task_dataset
        self.label_df = task_dataset.get_label_df()
        self.specific_assay = specific_assay
        self.sample_method = sample_method
        assert self.sample_method in ['stratify', 'random']

        # Extract list of assays
        self.assay_list = list(self.label_df['ASSAY'].unique())

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
                    support_set_df, query_set_df, label_support, label_query = task_sample(self.sample_method, chosen_assay_df, self.support_set_size, self.query_set_size)
                    """
                    if self.sample_method == 'stratify':
                        chosen_assay_df_2, support_set_df, _unused1, label_support = train_test_split(
                            chosen_assay_df, chosen_assay_df['LABEL'], test_size=self.support_set_size, stratify=chosen_assay_df['LABEL']
                        )
                        _unused_2, query_set_df, _unused3, label_query = train_test_split(
                            chosen_assay_df_2, chosen_assay_df_2['LABEL'], test_size=self.query_set_size, stratify=chosen_assay_df_2['LABEL']
                        )
                    elif self.sample_method == 'random':
                        chosen_assay_df_2, support_set_df, _unused1, label_support = train_test_split(
                            chosen_assay_df, chosen_assay_df['LABEL'], test_size=self.support_set_size, stratify=None
                        )
                        _unused_2, query_set_df, _unused3, label_query = train_test_split(
                            chosen_assay_df_2, chosen_assay_df_2['LABEL'], test_size=self.query_set_size, stratify=None
                        )
                    """
                    assert len(support_set_df) == self.support_set_size
                    assert len(query_set_df) == self.query_set_size
                    assert len(set(label_support)) == 2
                    assert len(set(label_query)) == 2
                    list_data_idx = list(support_set_df.index) + list(query_set_df.index) 
                except:
                    if self.specific_assay:
                        raise ValueError('Choose another assay that has more datapoints')
                else: 
                    bad_task = False
            
            # Yield a list, each element correspond to a row in the df
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