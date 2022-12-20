import os
import json

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.modules import Linear
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule.multitask_cp import multitask_pretrain_cp_dataset, load_FNN_with_trained_weights
from utils.models.multitask import cp_multitask
from utils.metrics import multitask_bce


def main():

    ### Inits
    num_repeat = 10
    support_set_sizes = [16, 32, 64, 96]
    query_set_size = 32
    max_epochs = 50
    loss_function = multitask_bce()
    devices = [2]
    #fraction_train_set = 0.9
    version = 0

    ### Paths inits
    data_folder = '/home/son.ha/FSL_CP/data/output'
    hparam_file = '/home/son.ha/FSL_CP/fsl_cp/hparams/fnn_multitask.json'
    cp_f_path = [os.path.join(data_folder, i) for i in [
        'norm_CP_feature_df.csv',
        #'norm_ECFP_feature_df.csv',
        #'norm_RDKit_feature_df.csv',
    ]]
    pretrain_logs_path = '/home/son.ha/FSL_CP/logs/multitask_only_cp_pretrain'
    path_to_weight = os.path.join(pretrain_logs_path, 'lightning_logs/version_0/checkpoints/final_model.ckpt')
    logs_path = '/home/son.ha/FSL_CP/logs/multitask_cp_finetune2'

    ### Final result dictionary
    result_before_pretrain = {
        'ASSAY_ID': [],
        '16_auc_before_train': [],
        '32_auc_before_train': [],
        '64_auc_before_train': [],
        '96_auc_before_train': []
    }

    final_result = {
        '16_auc_after_train': [],
        '32_auc_after_train': [],
        '64_auc_after_train': [],
        '96_auc_after_train': []
    }


   ### Load hparams from JSON files
    with open(hparam_file) as f:
            hparams = json.load(f)


    ### Load the assay keys
    with open(os.path.join(data_folder, 'data_split.json')) as f:
        data = json.load(f)
    test_split = data['test']
    result_before_pretrain['ASSAY_ID'] = test_split


    ### Loop through all support set sizes
    for support_set_size in support_set_sizes:
        for rep in range(num_repeat):

            ### Load data (support and query sets)
            support_set = multitask_pretrain_cp_dataset(
                assay_codes=test_split,
                label_df_path=os.path.join(data_folder, 'FINAL_LABEL_DF.csv'),
                cp_f_path=cp_f_path,
                set_size=support_set_size,
                inference=True,
                support=True,
                random_state=69
            )
            query_set = multitask_pretrain_cp_dataset(
                assay_codes=test_split,
                label_df_path=os.path.join(data_folder, 'FINAL_LABEL_DF.csv'),
                cp_f_path=cp_f_path,
                set_size=query_set_size,
                inference=True,
                support=False,
                random_state=69
            )
            #train_size = int(fraction_train_set * len(support_set))
            #val_size = len(support_set) - train_size
            #train_support_set, val_support_set = torch.utils.data.random_split(support_set, [train_size, val_size])


            ### Create DataLoader objects
            support_loader = DataLoader(support_set, batch_size=hparams["training"]["batch_size"],
                                    shuffle=False, num_workers=20)
            
            query_loader = DataLoader(query_set, batch_size=hparams["training"]["batch_size"],
                                    shuffle=False, num_workers=20)


            ### Load pretrained models
            fnn_pretrained = load_FNN_with_trained_weights(
                path_to_weight=path_to_weight,
                input_shape=len(support_set[3][0]),
            )
            fnn_pretrained.classifier[10] = Linear(in_features=2048, out_features=len(support_set[3][1]), bias=True)


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
                log_every_n_steps=20,
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

    return None

if __name__ == '__main__':
    main()

### TODO: time to open csv files are too long. Can do something to not repeat opening them again and again?