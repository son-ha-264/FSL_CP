import torch 
import torch.nn as nn
from utils.utils_proto_cp import JSONLBroadDataset, CPSampler, NormalizeByImage, accuracy, sliding_average, KlammbauerNetRelu
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch import optim
import json
import numpy as np
from sklearn.metrics import roc_auc_score



class ProtoNet(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(ProtoNet, self).__init__()
        self.backbone = backbone
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor
    ):
        """
        Predict query labels using labeled support images.
        """
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)
        n_way = 2
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels==label)].mean(0)
                for label in range(n_way)
            ]
        )

        dists = torch.cdist(z_query, z_proto)

        scores = -dists
        return scores

def fit(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        criterion, 
        optimizer,
        model
    ) -> float:
        optimizer.zero_grad()
        classification_scores = model(
            support_images.cuda(), support_labels.cuda(), query_images.cuda()
        )

        loss = criterion(classification_scores, query_labels.cuda())
        loss.backward()
        optimizer.step()

        return loss.item()

def evaluate_on_one_task(
    model,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
):
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
        torch.max(
            model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
            .detach()
            .data,
            1,
        )[1]).cpu().numpy(), query_labels.cpu().numpy()
    #    == query_labels.cuda()
    #).sum().item(), len(query_labels)


def evaluate(model, data_loader: DataLoader, support_size=64, save = False, plus=""):
    assert plus in ["", "+"]
    scores = []
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            y_pred, y_true = evaluate_on_one_task(
                model, support_images, support_labels, query_images, query_labels
            )
            score = roc_auc_score(y_true, y_pred)
            scores.append(score)
    if save:
        np.save("../output/output_protonet"+str(support_size)+plus+".npy", np.array(scores))

    print(
        f"Model tested on {len(data_loader)} episodes. Average ROC_AUC: {(np.mean(scores)):.2f}"
    )

def main():

    support_set_size = 64

    with open('/home/son.ha/FSL_CP/data/output/data_split.json') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    train_split = train_split + val_split
    
    train_data = JSONLBroadDataset(train_split, transform = None, cp_f_path=['/home/son.ha/FSL_CP/data/output/norm_CP_feature_df.csv'])
    test_data = JSONLBroadDataset(test_split, transform = None, cp_f_path=['/home/son.ha/FSL_CP/data/output/norm_CP_feature_df.csv'])

    #Pretraining
    train_sampler = CPSampler(
        task_dataset=train_data,
        support_set_size=support_set_size,
        query_set_size=64,
        tasks_per_batch=1024,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_data,
        batch_sampler=train_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    
    test_sampler = CPSampler(
        task_dataset=test_data,
        support_set_size=support_set_size,
        query_set_size=64,
        tasks_per_batch=200,
        shuffle=False,
        specific_assay='752347' #752347
    )
    test_loader = DataLoader(
        test_data,
        batch_sampler=test_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    input_shape=len(train_data[3][0])
    network = KlammbauerNetRelu(num_classes=1600, input_shape=input_shape)

    model = ProtoNet(network).cuda()

    print('Performance before training')
    evaluate(model, test_loader)

    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    log_update_frequency = 10
    
    all_loss = []
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            loss_value = fit(support_images, support_labels, query_images, query_labels, criterion, optimizer, model)
            all_loss.append(loss_value)

            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))


    print('Performance after training')
    evaluate(model, test_loader, support_size=support_set_size, save=False, plus="+")


if __name__ == '__main__':
    main()