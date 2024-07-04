import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base import BaseClassifier


class StatsAngleModel(BaseClassifier):

    def __init__(
            self,
            mean_pos_ratio: float=1.,
            mean_neg_ratio: float=1.,
            std_pos_ratio: float=1.,
            std_neg_ratio: float=1.,
            **kwargs
        ) -> None:

        super().__init__(**kwargs)

        self.stats_loss = StatsAngleLoss(
            mean_pos_ratio, mean_neg_ratio, std_pos_ratio, std_neg_ratio, self.n_classes
        )
        

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
            mean_pos_ratio: float,
            mean_neg_ratio: float,
            std_pos_ratio: float,
            std_neg_ratio: float,
            n_classes: int,
        ):

        super().__init__()

        self.mean_pos_ratio = mean_pos_ratio
        self.mean_neg_ratio = mean_neg_ratio
        self.std_pos_ratio = std_pos_ratio
        self.std_neg_ratio = std_neg_ratio
        self.n_classes = n_classes


    def forward(self, features, labels):
        
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape)==3:
            features = features.mean(dim=1)
        features = F.normalize(features, p=2, dim=1)

        labels = torch.argmax(labels, dim=1)

        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)
        mask_pos = mask.masked_fill(eye, 0).to(device)
        mask_neg = ~mask

        cos_dist = 1 - torch.matmul(features, features.t())
        cos_pos = cos_dist[mask_pos]
        cos_neg = cos_dist[mask_neg]

        mean_pos = self.mean_pos_ratio * torch.mean(cos_pos)**2
        mean_neg = self.mean_neg_ratio * (1-torch.mean(cos_neg))**2
        std_pos = self.std_pos_ratio * torch.std(cos_pos)
        std_neg = self.std_neg_ratio * torch.std(cos_neg)
        
        return mean_pos + mean_neg + std_pos + std_neg