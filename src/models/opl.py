import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base import BaseClassifier


class OPLModel(BaseClassifier):

    def __init__(
            self,
            gamma: float=2., 
            opl_ratio: float=0.1,
            **kwargs
        ) -> None:

        super().__init__(**kwargs)

        self.opl_loss = OrthogonalProjectionLoss(gamma, opl_ratio)
        

    def calc_loss(
            self, 
            features: Tensor,
            logits: Tensor,
            labels: Tensor, 
            **kwargs,
        ) -> Tensor:

        return self.ce_loss(logits, labels) + self.opl_loss(features, labels)
    


class OrthogonalProjectionLoss(nn.Module): 
    
    def __init__(
            self, 
            gamma: float=2., 
            opl_ratio: float=0.1
        ):

        super(OrthogonalProjectionLoss, self).__init__()

        self.weights_dict = None
        self.gamma = gamma
        self.opl_ratio = opl_ratio


    def forward(self, features, labels):
        
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape)==3:
            features = features.mean(dim=1)
        features = F.normalize(features, p=2, dim=1)

        labels = torch.argmax(labels, dim=1)

        labels = labels[:, None]  # extend dim
        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (self.gamma * neg_pairs_mean)

        return self.opl_ratio * loss