from typing import Dict, Union, Optional, Tuple
import os 

import torch
from torch import nn

from src.models import *
from src.helpers.training import read_config, save_config, get_models_list


__all__ = [
    "get_model",
    "read_configs_model",
    "load_epoch_model", 
    "load_last_epoch_model",
]


def get_model(
        device: torch.device,
        n_classes: int,
        cfg_encoder: Dict,
        cfg_classifier: Dict,
        parallel: bool=True,
    ) -> Union[BaseClassifier, ClassifierParallel]:

    model = get_model_type(
        n_classes=n_classes,
        cfg_encoder=cfg_encoder,
        **cfg_classifier,
    ).to(device=device, dtype=torch.float)
    return ClassifierParallel(model) if parallel else model


def read_configs_model(
        path,
        config_base: str, 
        config_classifier: str,
        results_dir: str="/", 
        save: bool=False,
    ) -> Tuple[Dict, Dict, Dict, Dict]:

    cfg_base = read_config(os.path.join(path, "base", config_base))
    cfg_classifier = read_config(os.path.join(path, "classifiers", config_classifier))
    cfg_data = cfg_base['data']
    cfg_encoder = cfg_base['encoder']
    cfg_optimizer = cfg_base['optimizer']
    cfg_scheduler = cfg_base['scheduler']

    if save:
        save_config(cfg_base, os.path.join(results_dir,"config_base"))
        save_config(cfg_classifier, os.path.join(results_dir,"config_classfier"))

    return cfg_data, cfg_encoder, cfg_classifier, cfg_optimizer, cfg_scheduler
   

def load_last_epoch_model(
        model_dir: str,
        model: Union[nn.Module, ClassifierParallel],
        optimizer: torch.optim.Optimizer,
        restart: bool=False
    ) -> int:

    previous_models = get_models_list(model_dir, 'epoch_')
    if len(previous_models)>0 and not restart:
        checkpoint = torch.load(os.path.join(model_dir, previous_models[-1]))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']
    else:
        return 0


def load_epoch_model(
        epoch: int,
        device: torch.device,
        model_dir: str,
        model: Optional[Union[nn.Module, ClassifierParallel]]=None,
        optimizer: Optional[torch.optim.Optimizer]=None,
    ) -> None:

    epoch_path = os.path.join(model_dir, ("epoch_"+str(epoch)+".pt"))
    checkpoint = torch.load(epoch_path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])


def save_epoch_model(
        epoch: int,
        model_dir: str,
        model: Union[nn.Module, ClassifierParallel],
        optimizer: torch.optim.Optimizer,
    ) -> None:

    epoch_path = os.path.join(model_dir, ("epoch_"+str(epoch)+".pt"))
    model_state_dict = model.module.state_dict() if \
        isinstance(model, ClassifierParallel) else model.state_dict()
    checkpoint = {
        'epoch':        epoch,
        'model':        model_state_dict,
        'optimizer':    optimizer.state_dict(),
    }
    torch.save(checkpoint, epoch_path)