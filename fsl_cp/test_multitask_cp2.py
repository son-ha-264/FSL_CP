import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0'

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.modules import Linear
import torch.optim as optim
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule.multitask_cp import multitask_pretrain_cp_dataset, load_FNN_with_trained_weights
from utils.models.multitask import cp_multitask
from utils.metrics import multitask_bce


def main():

    ### Inits
    num_repeat = 100
    support_set_sizes = [16, 32, 64, 96]
    query_set_size = 32
    max_epochs = 50
    loss_function = multitask_bce()
    sigmoid = torch.nn.Sigmoid()
    
    #devices = [2]
    #fraction_train_set = 0.9
    #version = 0

    ### Paths inits
    HOME = os.environ['HOME']
    data_folder = os.path.join(HOME, 'FSL_CP/data/output')
    df_assay_id_map_path = os.path.join(HOME, 'FSL_CP/data/output/assay_target_map.csv') 
    result_summary_path = os.path.join(HOME, 'FSL_CP/result/result_summary/multitask_cp_result_summary.csv') 

    ### Open feature df
    feature_df = pd.read_csv(os.path.join(data_folder, 'norm_CP_feature_df.csv'))
    feature_df = feature_df.dropna(axis=1, how='any')
    feature_df = feature_df.drop(columns=['INCHIKEY', 'CPD_SMILES', 'SAMPLE_KEY'])

    pretrain_logs_path = '/home/son.ha/FSL_CP/logs/multitask_only_cp_pretrain'
    path_to_weight = os.path.join(pretrain_logs_path, 'lightning_logs/version_0/checkpoints/final_model.ckpt')
    logs_path = '/home/son.ha/FSL_CP/logs/multitask_cp_finetune2'

    ### Final result dictionary
    final_result = {
        '16': [],
        '32': [],
        '64': [],
        '96': []
    }


    ### Load the assay keys
    with open(os.path.join(data_folder, 'data_split.json')) as f:
        data = json.load(f)
    test_split = data['test']
    final_result['ASSAY_ID'] = test_split


    ### Loop through all support set sizes
    for support_set_size in tqdm(support_set_sizes, desc='Support set size'):
        for test_assay in tqdm(test_split, desc='Test split'):
            test_assay = [test_assay]
            list_auc = []
            for rep in range(num_repeat):

                ### Load data (support and query sets)
                support_set = multitask_pretrain_cp_dataset(
                    assay_codes=test_assay,
                    label_df_path=os.path.join(data_folder, 'FINAL_LABEL_DF.csv'),
                    feature_df=feature_df,
                    set_size=support_set_size,
                    inference=True,
                    support=True,
                    random_state=None
                )
                query_set = multitask_pretrain_cp_dataset(
                    assay_codes=test_assay,
                    label_df_path=os.path.join(data_folder, 'FINAL_LABEL_DF.csv'),
                    feature_df=feature_df,
                    set_size=query_set_size,
                    inference=True,
                    support=False,
                    random_state=None
                )
                #train_size = int(fraction_train_set * len(support_set))
                #val_size = len(support_set) - train_size
                #train_support_set, val_support_set = torch.utils.data.random_split(support_set, [train_size, val_size])


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
                fnn_pretrained.classifier[10] = Linear(in_features=2048, out_features=len(support_set[3][1]), bias=True)
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
            print(list_auc)
            final_result[str(support_set_size)].append(f"{np.mean(list_auc):.2f}+/-{np.std(list_auc):.2f}")

    ### Create result summary dataframe
    df_assay_id_map = pd.read_csv(df_assay_id_map_path)
    df_assay_id_map = df_assay_id_map.astype({'ASSAY_ID': str})
    df_score = pd.DataFrame(data=final_result)
    df_final = pd.merge(df_assay_id_map[['ASSAY_ID', 'assay_chembl_id']], df_score, on='ASSAY_ID', how='right')
    df_final.to_csv(result_summary_path, index=False)
    '''
    ### Fine-tune on support set
    train_loop = cp_multitask(
        fnn_pretrained, loss_function,
        lr=hparams["optimizer_params"]["lr"],
        momentum=hparams["optimizer_params"]["momentum"],
        weight_decay=hparams["optimizer_params"]["weight_decay"],
        step_size=hparams["lr_schedule"]["step_size"],
        gamma=hparams["lr_schedule"]["gamma"],
    )
    #checkpoint_callback = ModelCheckpoint(mode="min", monitor="val_loss", save_top_k = 1)
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=devices, 
        max_epochs=max_epochs,
        gradient_clip_val=hparams["grad_clip"]["val"], 
        gradient_clip_algorithm=hparams["grad_clip"]["algorithm"],
        strategy=DDPStrategy(find_unused_parameters=False),
        #log_every_n_steps=20,
        #callbacks=[
        #    checkpoint_callback
        #],
        enable_checkpointing=False,
        logger=False
    )
    trainer.fit(train_loop, support_loader)
    final_model_path = os.path.join(
        logs_path,
        "lightning_logs/",
        "version_"+str(version),
        "checkpoints",
        "final_model.ckpt"
    )
    trainer.save_checkpoint(final_model_path)


    ### Predict
    best_model = cp_multitask.load_from_checkpoint(final_model_path, 
                                            **{"model": fnn_pretrained, "loss_function": multitask_bce()})
    trainer2 = pl.Trainer(accelerator="gpu", devices=devices, logger=False)
    predictions = trainer2.predict(best_model, dataloaders=query_loader)
    #torch.save(predictions, "/home/son.ha/FSL_CP/output/multitask_pred/multitask_"+str(support_set_size)+"_"+str(rep)+".pt")
    query_set.get_label_df().to_csv('/home/son.ha/FSL_CP/output/multitask_pred_3/query_set_'+str(support_set_size)+"_"+str(rep)+'.csv')
    torch.save(predictions, "/home/son.ha/FSL_CP/output/multitask_pred_3/multitask_pred_"+str(support_set_size)+"_"+str(rep)+".pt")
'''

    return None

if __name__ == '__main__':
    main()
