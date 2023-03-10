import os
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from os.path import expanduser
from torch.nn.modules import Linear
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

from utils.metrics import multitask_bce
from utils.misc import NormalizeByImage
from datamodule.multitask_img import prepare_support_query_multitask_img, load_CNN_with_trained_weights
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
        '-p', '--path_to_image', type=str, default='/mnt/scratch/Son_cellpainting/my_cp_images/',
        help='Path to folder of Cell Painting images')
    parser.add_argument(
        '-d', '--device', type=str, default='cuda:3',
        help='gpu or cpu')
    parser.add_argument(
        '-s', '--resize', type=int, default=520,
        help='Resize image to a square of this size.')
    parser.add_argument(
        '-c', '--checkpoint_path', type=str, default=None,
        help='path to ckpt file containing model weights.')
    args = parser.parse_args()

    image_path = args.path_to_image
    image_resize = args.resize
    path_to_weight = args.checkpoint_path
    device = args.device

    if device == 'cpu':
        device = torch.device('cpu')
    elif 'cuda' in device and torch.cuda.device_count():
        device = torch.device(device)

    ### Inits
    num_repeat = 100
    #step_size = 100
    support_set_sizes = [8, 16, 32, 64, 96]
    query_set_size = 32
    max_epochs = 50
    crop_size = 500
    image_resize = 500
    loss_function = nn.BCEWithLogitsLoss()
    sigmoid = torch.nn.Sigmoid()

    ### Define image transformation
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomCrop(crop_size),
        transforms.Resize((image_resize,image_resize)),
        NormalizeByImage()])  

    ### Paths inits
    HOME = expanduser("~")
    data_folder = os.path.join(HOME, 'FSL_CP/data/output')
    df_assay_id_map_path = os.path.join(HOME, 'FSL_CP/data/output/assay_target_map.csv') 
    result_summary_path1 = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_img_auroc_result_summary.csv') 
    result_summary_path2 = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_img_dauprc_result_summary.csv') 
    result_summary_path3 = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_img_bacc_result_summary.csv') 
    result_summary_path4 = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_img_f1_result_summary.csv') 
    result_summary_path5 = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_img_kappa_result_summary.csv') 

    if not path_to_weight:
        path_to_weight = os.path.join(HOME, 'FSL_CP/weights/multitask_img/final_model.ckpt')

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
                support_set, query_set = prepare_support_query_multitask_img(
                    assay_code=test_assay,
                    label_df_path=os.path.join(data_folder, 'FINAL_LABEL_DF.csv'),
                    support_set_size=support_set_size,
                    query_set_size=query_set_size,
                    image_path=image_path,
                    transform=transform
                )


                ### Create DataLoader objects
                support_loader = DataLoader(support_set, batch_size=12,
                                        shuffle=True, num_workers=20)
                
                query_loader = DataLoader(query_set, batch_size=12,
                                        shuffle=False, num_workers=20)

                
                ### Load pretrained models
                cnn_pretrained = load_CNN_with_trained_weights(
                    path_to_weight=path_to_weight,
                    input_shape=len(support_set[3][0]),
                )
                num_ftrs = cnn_pretrained.fc.in_features
                cnn_pretrained.fc = nn.Linear(num_ftrs, 1, bias=True)
                cnn_pretrained = cnn_pretrained.to(device)

                # Fine-tune
                #optimizer = optim.SGD(fnn_pretrained.parameters(), lr=0.001, momentum=0.9)
                optimizer = optim.Adam(cnn_pretrained.parameters(), 1e-3)
                #scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
                with tqdm(range(max_epochs), total=max_epochs, leave=False, desc='Training on support set') as tqdm_train:
                    for epoch in tqdm_train:
                        all_loss = []
                        for i, (inputs, labels) in enumerate(support_loader, 0):
                            optimizer.zero_grad()
                            outputs = cnn_pretrained(inputs.to(device))
                            loss = loss_function(torch.squeeze(outputs), labels.to(device))
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(cnn_pretrained.parameters(), max_norm=5)
                            optimizer.step()
                            #if epoch % step_size == 0 and epoch!=0:
                            #scheduler.step()
                            all_loss.append(loss.item())
                        tqdm_train.set_postfix(train_loss=np.mean(all_loss))

                # Inference
                cnn_pretrained.eval()
                list_pred = []
                list_true = []
                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(query_loader, 0):
                        pred = cnn_pretrained(inputs.to(device))
                        pred = sigmoid(pred)
                        list_pred.append(pred)
                        list_true.append(labels)
                tensor_pred = torch.cat(list_pred, 0)
                tensor_pred = torch.squeeze(tensor_pred)
                pred_array = tensor_pred.cpu().detach().numpy()
                tensor_true = torch.cat(list_true, 0)
                true_array = tensor_true.cpu().detach().numpy()
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