import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base import BaseClassifier


class StatsAngleModel(BaseClassifier):

    def __init__(
            self,
            mean_lambda: float=1.,
            std_lambda: float=1.,
            **kwargs
        ) -> None:

        super().__init__(**kwargs)

        self.stats_loss = StatsAngleLoss(mean_lambda, std_lambda, self.n_classes)
        

    def calc_loss(
            self, 
            features: Tensor,
            logits: Tensor,
            labels: Tensor, 
            **kwargs,
        ) -> Tensor:

        return self.ce_loss(logits, labels) + self.stats_loss(features, labels)
    


class StatsAngleLoss(nn.Module): 
    
    def __init__(
            self, 
            mean_lambda: float,
            std_lambda: float,
            n_classes: int,
        ):

        super().__init__()

        self.mean_lambda = mean_lambda
        self.std_lambda = std_lambda
        self.n_classes = n_classes


    def forward(self, features, labels):

        if len(features.shape)==3:
            features = features.mean(dim=1)
        features = F.normalize(features, p=2, dim=1)

        if len(labels.shape)==1:
            labels = F.one_hot(labels, self.n_classes)
            
        labels = F.normalize(labels.to(dtype=torch.float), p=2, dim=1)
        labels_dist = 1 - torch.matmul(labels, labels.t())
        embeds_dist = 1 - torch.matmul(features, features.t())
        mask_pos = labels_dist<1e-3
        mask_neg = labels_dist>1-1e-3
        mask_pos.fill_diagonal_(False)
        mask_neg.fill_diagonal_(False)
        dif = (embeds_dist - labels_dist)**2
        dif.fill_diagonal_(0)

        loss_mean = dif.mean()
        loss_std = (dif[mask_pos]).std()# + (dif[mask_neg]).std()

        return self.mean_lambda * loss_mean# + self.std_lambda * loss_std