from typing import List, Tuple
import random
import torch
import jsonlines
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch import Tensor
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class JSONLBroadDataset(Dataset):
    """DataLoader for JSONL files. 
       Input JSONL files, convert them to lists of imputs and labels

    Args:
        assay_code(List[str]): list of code numbers of assay
        img_folder(str): path to image folder

    """
    def __init__(self, 
                assay_codes: List[str], 
                img_folder='/mnt/scratch/Son_cellpainting/my_cp_images/',
                transform=None,
                data_folder='/home/son.ha/FSL_CP/data/jsonl/',
                cp_f_path = [
                    '/home/son.ha/FSL_CP/data/output/norm_CP_feature_df.csv',
                    '/home/son.ha/FSL_CP/data/output/norm_ECFP_feature_df.csv',
                    '/home/son.ha/FSL_CP/data/output/norm_RDKit_feature_df.csv',
                ],
    ):
        super(JSONLBroadDataset).__init__()

        # Inits
        self.transform = transform
        self.num_row_cp_f = []
        self.labels = []
        self.inchi = []
        self.assay = []
        self.sample_key = []
        self.assay_codes = assay_codes
        self.img_folder = img_folder

        # Load JSONL files into an input and output lists for future batching
        for assay_code in self.assay_codes:
            jsonl_path = os.path.join(data_folder, assay_code+'.jsonl')
            with jsonlines.open(jsonl_path) as reader:
                for obj in reader:
                    self.num_row_cp_f.append(obj['NUM_ROW_CP_FEATURES'])
                    self.labels.append(obj['LABEL'])
                    self.inchi.append(obj['INCHIKEY'])
                    self.assay.append(assay_code)
                    self.sample_key.append(obj['SAMPLE_KEY'])
        
        # Read feature matrices and concat them
        list_feature_df = []
        for path in cp_f_path:
            feature_df = pd.read_csv(path)
            feature_df = feature_df.drop(columns=['INCHIKEY', 'CPD_SMILES', 'SAMPLE_KEY'])
            list_feature_df.append(feature_df)
        self.feature_df = pd.concat(list_feature_df, axis=1)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        cp_f_df_idx = self.num_row_cp_f[idx]
        x = torch.tensor(self.feature_df.iloc[cp_f_df_idx, :],dtype=torch.float)
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x,y
    
    def get_labels(self):
        """ Return a list of all labels for sampling
        """
        return self.labels
    
    def get_assay(self, unique=False):
        """Return a list of all assays
        """
        if unique:
            return(list(set(self.assay)))
        else:
            return self.assay


def _sample_with_seed(seq, n, seed):
    if seed>=0:
        random.seed(seed)
    ans = random.sample(seq, n)
    return ans


class CPSampler(Sampler):
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
        task_dataset: JSONLBroadDataset,
        support_set_size: int,
        query_set_size: int,
        tasks_per_batch: int,
        shuffle = False,
        specific_assay = None,
        seed = -1
    ):
        super().__init__(data_source=None)
        self.n_shots_support = int(support_set_size/2) #NOTE!: here is actually n-shots!!!, support_set = 2 x n_shots 
        self.n_shots_query = int(query_set_size/2)
        self.tasks_per_batch = tasks_per_batch
        self.task_dataset = task_dataset
        self.shuffle = shuffle
        self.assay_unique = task_dataset.get_assay(unique=True)
        self.assays = task_dataset.get_assay()
        self.labels = task_dataset.get_labels()
        self.specific_assay = specific_assay
        self.seed = seed

    def __len__(self):
        return self.tasks_per_batch
        
    def __iter__(self):
        for _ in range(self.tasks_per_batch):
            
            #sample one task/assay 
            bad_task = True
            while bad_task: 
                if self.specific_assay:
                    sampled_task = self.specific_assay
                else:
                    sampled_task = random.sample(self.assay_unique, 1)[0]
                items_per_label = {}
                for item, assay_label in enumerate(zip(self.assays, self.labels)):
                    if assay_label[0]!=sampled_task:
                        continue
                    else:
                        if assay_label[1] in items_per_label.keys():
                            items_per_label[assay_label[1]].append(item)
                        else:
                            items_per_label[assay_label[1]] = [item]
                try:
                    ans = [
                    torch.tensor(
                        _sample_with_seed(
                            items_per_label[label], self.n_shots_support + self.n_shots_query, seed=self.seed
                        )
                    )
                    for label in items_per_label
                ]
                except:
                    if self.specific_assay:
                        raise ValueError('Choose another assay that has more datapoints')
                else:
                    bad_task = False
            yield torch.cat(
                ans                
            ).tolist()

    def episodic_collate_fn(
        self,
        input_data: List[Tuple[Tensor, int]]
    ):
        if self.shuffle:
            if self.seed >= 0:
                random.seed(self.seed)
            random.shuffle(input_data)

        true_class_ids = list({x[1] for x in input_data})

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (2, self.n_shots_support + self.n_shots_query, *all_images.shape[1:])
        )
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((2, self.n_shots_support + self.n_shots_query))

        support_images = all_images[:, : self.n_shots_support].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shots_support :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shots_support].flatten()
        query_labels = all_labels[:, self.n_shots_support :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )

def _multitask_bce(pred, target):
    """Function to calculate multitask bce. Average bce across tasks"""
    eps = 1e-7
    mask = (target != -1).float().detach() # not-1->1, -1->0
    bce = pred.clamp(min=0) - pred*target + torch.log(1.0 + torch.exp(-pred.abs()))
    bce[mask == 0] = 0
    loss = bce.sum() / (mask.sum() + eps)
    return(loss)


class multitask_bce(torch.nn.Module):
    """Class wrapper of the _multitask_bce function"""
    def __init__(self):
        super(multitask_bce, self).__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return(_multitask_bce(pred, target))

def accuracy(prediction, target):
    """Computes the precision@k for the specified values of k"""
    mask = (target != -1)
    acc = ((target == prediction.round()) * mask).sum() / mask.sum()
    return acc

class NormalizeByImage(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

    Credit: https://github.com/ml-jku/hti-cnn/blob/master/pyll/transforms.py
    """
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        for t in tensor:
            t.sub_(t.mean()).div_(t.std() + 1e-7)
        return tensor

def sliding_average(value_list: List[float], window: int) -> float:
    """
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats (can't be empty)
        window: number of instances to take into account. If value is 0 or greater than
            the length of value_list, all instances will be taken into account.
    Returns:
        average of the last window instances in value_list
    Raises:
        ValueError: if the input list is empty
    """
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()

class KlammbauerNetRelu(nn.Module):
    def __init__(self, model_params=None, num_classes=209, input_shape=None):
        super(KlammbauerNetRelu, self).__init__()
        assert input_shape
        in_d = input_shape
        fc_units = 2048
        drop_prob = 0.5
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_d, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, num_classes),
            #nn.Flatten()
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