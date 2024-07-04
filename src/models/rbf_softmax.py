import torch
from torch import nn

from .base import BaseClassifier


class RBFModel(BaseClassifier):

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        self.class_head = RBFLogits(self.embed_dim, self.n_classes)
    


class RBFLogits(nn.Module):

    def __init__(
            self, 
            embed_dim: int, 
            n_classes: int,
        ):

        super(RBFLogits, self).__init__()
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, embed_dim))
        self.bias = nn.Parameter(torch.FloatTensor(n_classes))
        self.scale = 4 if n_classes==10 else 14
        self.gamma = 1.8 if n_classes==10 else 2.2
        nn.init.xavier_uniform_(self.weight)


    def forward(self, embeds):

        diff = torch.unsqueeze(self.weight, dim=0) - torch.unsqueeze(embeds, dim=1)
        diff = torch.mul(diff, diff)
        metric = torch.sum(diff, dim=-1)
        kernal_metric = torch.exp(-1.0 * metric / self.gamma)
        if self.training:
            train_logits = self.scale * kernal_metric
            return train_logits
        else:
            test_logits = self.scale * kernal_metric
            return test_logits