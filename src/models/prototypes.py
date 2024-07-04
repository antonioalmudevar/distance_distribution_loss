from scipy.stats import ortho_group
import torch
from torch import nn, Tensor

from .base import BaseClassifier
from ..utils import jl_transform


class PredefinedPrototypesModel(BaseClassifier):

    def __init__(
            self,
            proto_lambda: float=1.,
            proto_scale: float=None,
            **kwargs
        ) -> None:

        super().__init__(**kwargs)

        self.proto_lambda = proto_lambda
        self.proto_loss = PredefinedPrototypesLoss(
            self.n_classes, self.embed_dim, proto_scale
        )
        

    def calc_loss(
            self, 
            features: Tensor,
            logits: Tensor,
            labels: Tensor, 
            **kwargs,
        ) -> Tensor:
        ce_loss = self.ce_loss(logits, labels)
        proto_loss = self.proto_loss(features, labels)
        return  ce_loss + self.proto_lambda * proto_loss



class PredefinedPrototypesLoss(nn.Module):

    def __init__(
            self,
            n_classes: int,
            embed_dim: int,
            proto_scale: float=None,
        ) -> None:
        super().__init__()

        if embed_dim >= n_classes:
            mean_class = ortho_group.rvs(dim=embed_dim)[:n_classes]
        else:
            mean_class = ortho_group.rvs(dim=n_classes)
            mean_class = jl_transform(mean_class, embed_dim)
        self.register_buffer('mean_class', torch.from_numpy(mean_class))

        if proto_scale:
            self.register_buffer('proto_scale', torch.Tensor([proto_scale]))
        else:
            self.register_buffer('proto_scale', nn.Parameter(0.1*torch.randn(1)))


    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        if len(features.shape)==3:
            features = features.mean(dim=1)
        proto_y = torch.abs(self.proto_scale)*torch.matmul(labels, self.mean_class)
        return torch.mean(torch.mean( (features - proto_y)**2 , axis=1))