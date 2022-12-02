import torch.nn as nn
import torch


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
        Predict query labels using support images with labels.
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