from typing import Dict

from .base import BaseClassifier, ClassifierParallel
from .opl import OPLModel
from .rbf_softmax import RBFModel
from .focal_loss import FocalLossModel
from .prototypes import PredefinedPrototypesModel
from .prototypes_ang import PredefinedPrototypesAngularModel
from .stats_angle import StatsAngleModel
from .encoders import get_encoder

__all__ = [
    "get_model_type",
    "BaseClassifier",
    "ClassifierParallel",
]


MODELS = {
    'CE':               BaseClassifier,
    'OPL':              OPLModel,
    'RBF':              RBFModel,
    'FOCAL_LOSS':       FocalLossModel, 
    'PROTOTYPES':       PredefinedPrototypesModel,   
    'PROTOTYPES_ANG':   PredefinedPrototypesAngularModel, 
    'STATS_ANGLE':      StatsAngleModel,
}


def get_model_type(
        cfg_encoder: Dict, 
        model_type: str,
        **kwargs
    ) -> BaseClassifier:

    encoder = get_encoder(**cfg_encoder)
    return MODELS[model_type.upper()](encoder=encoder, **kwargs)