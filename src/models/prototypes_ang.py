from scipy.stats import ortho_group
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base import BaseClassifier
from ..utils import jl_transform


class PredefinedPrototypesAngularModel(BaseClassifier):

    def __init__(
            self,
            proto_lambda: float=1.,
            **kwargs
        ) -> None:

        super().__init__(**kwargs)

        self.proto_lambda = proto_lambda
        self.proto_loss = PredefinedPrototypesAngularLoss(
            self.n_classes, self.embed_dim
        )
        

    def calc_loss(
            self, 
            features: Tensor,
            logits: Tensor,
            labels: Tensor, 
            **kwargs,
        ) -> Tensor:

        return self.ce_loss(logits, labels) + self.proto_lambda * self.proto_loss(features, labels)



class PredefinedPrototypesAngularLoss(nn.Module):

    def __init__(
            self,
            n_classes: int,
            embed_dim: int,
        ) -> None:
        super().__init__()

        if embed_dim >= n_classes:
            mean_class = ortho_group.rvs(dim=embed_dim)[:n_classes]
        else:
            mean_class = ortho_group.rvs(dim=n_classes)
            mean_class = jl_transform(mean_class, embed_dim)
        self.register_buffer('mean_class', torch.from_numpy(mean_class))


    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        if len(features.shape)==3:
            features = features.mean(dim=1)
        proto_y = torch.matmul(labels, self.mean_class)
        features = F.normalize(features, p=2, dim=1)
        proto_y = F.normalize(proto_y, p=2, dim=1)
        return torch.diag(1 - features @ proto_y.T).mean()