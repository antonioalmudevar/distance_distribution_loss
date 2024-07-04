import os
from typing import List, Optional, Dict, Any, Tuple
import logging
import logging.handlers
import math

import yaml
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#======Data Loaders=========================================================
NUM_WORKERS = 4

def get_loader(
        dataset: Dataset,
        batch_size: int=1, 
        shuffle: bool=False,
        num_workers: int=NUM_WORKERS, 
        pin_memory: bool=True,
    ):

    loader = DataLoader(
        dataset=dataset, 
        shuffle=shuffle, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    n_iters = int(np.ceil(len(dataset) / batch_size)) # type: ignore

    return loader, n_iters


#======Optimizer & Scheduler=========================================================
OPTIMIZERS = {
    'SGD':      optim.SGD,
    'ADAM':     optim.Adam,
    'ADAMW':    optim.AdamW,
}

def get_optimizer(
        params, 
        optimizer: str, 
        lr: float, 
        batch_size: Optional[int], 
        **kwargs
    ) -> Dict[str, Any]:
    assert optimizer.upper() in OPTIMIZERS, "optimizer is not correct"
    return {
        'optimizer': OPTIMIZERS[optimizer.upper()](params, lr=lr, **kwargs), 
        'learning_rate': lr,
    }


class Scheduler():

    def __init__(
            self,
            optimizer: optim.Optimizer,
            total_batches: int,
            learning_rate: float=0.5,
            epochs: int=500,
            lr_decay_rate: float=0.1,
            lr_decay_epochs: List[int]=[100, 150, 180],
            cosine: bool=False,
            warm_epochs: bool=0,
            warmup_from: float=0.00001
        ) -> None:

        self.warm_epochs = warm_epochs
        if warm_epochs>0:
            self.warmup_from = warmup_from
            if cosine:
                eta_min = learning_rate * (lr_decay_rate ** 3)
                self.warmup_to = eta_min + (learning_rate - eta_min) * (
                        1 + math.cos(math.pi * self.warm_epochs / epochs)) / 2
            else:
                self.warmup_to = learning_rate

        self.optimizer = optimizer
        self.n_epochs = epochs
        self.total_batches = total_batches
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epochs = lr_decay_epochs
        self.cosine = cosine


    def adjust_learning_rate(self, epoch):

        lr = self.learning_rate
        if self.cosine:
            eta_min = lr * (self.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / self.n_epochs)) / 2
        else:
            steps = np.sum(epoch > np.asarray(self.lr_decay_epochs))
            if steps > 0:
                lr = lr * (self.lr_decay_rate ** steps)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    
    def warmup_learning_rate(self, epoch: int, batch_id: int):

        if self.warm_epochs and epoch <= self.warm_epochs:
            p = (batch_id + (epoch - 1) * self.total_batches) / (self.warm_epochs * self.total_batches)
            lr = self.warmup_from + p * (self.warmup_to - self.warmup_from)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    
    def state_dict(self):
        return None


def get_optimizer_scheduler(
        model: nn.Module, 
        total_batches: int, 
        cfg_optimizer: Dict[str, Any], 
        cfg_scheduler: Dict[str, Any],
    ) -> Tuple[optim.Optimizer, Scheduler, int]:
    optimizer = get_optimizer(model.parameters(), **cfg_optimizer)
    scheduler = Scheduler(total_batches=total_batches, **optimizer, **cfg_scheduler)
    return optimizer['optimizer'], scheduler, scheduler.n_epochs


#======Setup Loggers=========================================================
class FormatterNoInfo(logging.Formatter):

    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path='', restart=False):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        if restart: 
            try:
                os.remove(log_path)
            except OSError:
                pass
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=(1024 ** 2 * 2), backupCount=3
        )
        file_formatter = logging.Formatter("%(asctime)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)


#======Count paramters=========================================================
def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#======Configs / Directories=========================================================
def read_config(path: str) -> Dict:
    with open(str(path)+".yaml", 'r') as f:
        return yaml.load(f, yaml.FullLoader)


def save_config(cfg, path: str) -> None:
    with open(str(path)+".yaml", 'w') as f:
        return yaml.dump(cfg, f)
    
def create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

#======Get list of models=========================================================
def get_models_list(dir: str, prefix: str):
    models = [epoch for epoch in os.listdir(dir) if epoch.startswith(prefix)]
    models_int = sorted([int(epoch[len(prefix):-3]) for epoch in models])
    return [prefix+str(epoch)+'.pt' for epoch in models_int]