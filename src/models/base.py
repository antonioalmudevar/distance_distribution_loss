from typing import Dict, OrderedDict, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel

from src.models.encoders import *


class BCELoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss(**kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.bce(self.sigmoid(input), target)



class BaseClassifier(nn.Module):

    def __init__(
            self,
            encoder: Union[PretrainedASTModel, BEATs, CRNN],
            n_classes: int,
            temperature: float=1.,
            freeze_encoder: bool=False,
            soft_labels: bool=False,
        ) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.temperature = temperature
        self.freeze_encoder = freeze_encoder
        self.embed_dim = encoder.embed_dim
        self.soft_labels = soft_labels

        self.class_head = nn.Linear(self.embed_dim, self.n_classes)
        self.ce_loss = BCELoss() if soft_labels else nn.CrossEntropyLoss()


    def _get_features(self, x: Tensor) -> Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        return features
    

    def _get_logits(self, features: Tensor) -> Tensor:
        logits = self.class_head(features)
        if len(logits.shape)==3:
            logits = logits.mean(dim=1)
        return logits
    

    def _get_preds(self, logits: Tensor) -> Tensor:
        preds = (logits / self.temperature)
        if self.soft_labels:
            return preds.sigmoid()
        else:
            return preds.softmax(dim=-1)
        

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self._get_features(x)
        logits = self._get_logits(features)
        preds = self._get_preds(logits)
        return {
            'features':         features, 
            'norm_features':    F.normalize(features, dim=1),
            'logits':           logits,
            'preds':            preds
        }
    

    def calc_loss(
            self, 
            logits: Tensor,
            labels: Tensor, 
            **kwargs,
        ) -> Tensor:
        return self.ce_loss(logits, labels)


class ClassifierParallel(DataParallel):

    def __init__(self, module: BaseClassifier, *args, **kwargs) -> None:
        super().__init__(module, *args, **kwargs)
        self.embed_dim = module.embed_dim

    def load_state_dict(
            self,
            state_dict: OrderedDict[str, Tensor],
            strict: bool = True,
        ):
        return self.module.load_state_dict(state_dict, strict)

    def state_dict(self):
        return self.module.state_dict()

    def calc_loss(self, *args, **kwargs) -> Tensor:
        return self.module.calc_loss(*args, **kwargs) # type: ignore