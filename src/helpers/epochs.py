from typing import Dict, Any, Union, Optional
from logging import Logger
import os
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import BaseClassifier, ClassifierParallel
from src.utils import *
from .training import Scheduler, create_directory
from .models import save_epoch_model

__all__ = [
    "train_epoch",
    "test_epoch",
]


#==========TRAIN EPOCHS====================
def train_epoch(
        epoch: int, 
        device: torch.device,
        train_loader: DataLoader,
        model: Union[BaseClassifier, ClassifierParallel],
        optimizer: torch.optim.Optimizer,
        scheduler: Scheduler,
        model_dir: Optional[str]=None,
    ):

    model.train()

    scheduler.adjust_learning_rate(epoch)

    for iter, (input, labels) in enumerate(train_loader, start=1):

        scheduler.warmup_learning_rate(epoch, iter)

        input = input.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        #======Forward=======
        output = model(input)
        loss = torch.mean(model.calc_loss(**output, labels=labels))

        #======Backward=======
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if model_dir is not None:
        save_epoch_model(epoch, model_dir, model, optimizer)



#==========TEST EPOCH====================
def test_epoch(
        logger: Logger,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: Union[BaseClassifier, ClassifierParallel],
        preds_dir: Optional[str]=None,
    ):
    
    model.eval()
    outputs_train = get_outputs(device, train_loader, model, True)
    outputs_test = get_outputs(device, test_loader, model, True)

    #==========CLASSIFICATION METRICS===============
    metrics_classification = {
        'Train':    calc_metrics_classification(outputs_train),
        'Test':     calc_metrics_classification(outputs_test)
    }
    logger.info(formatted_2_dict(metrics_classification))

    if preds_dir:
        create_directory(preds_dir)
        save_metrics(preds_dir, metrics_classification)
        save_outpus(preds_dir, 'train', outputs_train)
        save_outpus(preds_dir, 'test', outputs_test)


def get_outputs(
        device: torch.device,
        loader: DataLoader,
        model: Union[BaseClassifier, ClassifierParallel],
        calc_loss: bool=False,
    ) -> Dict[str, Any]:

    n_iters, loss, labels = 0, 0, []
    outputs = {'features': [], 'norm_features': [], 'logits': [], 'preds': []}

    for input, target in loader:

        n_iters += 1
        labels.extend(target)
        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        with torch.no_grad():
            output = model(input)
            for key in outputs:
                outputs[key].extend(output[key].cpu().detach())
            if calc_loss:
                loss += torch.mean(model.calc_loss(**output, labels=target)).item()

    outputs_final = {
        key: torch.stack(value).numpy() for key, value in outputs.items()
    }
    outputs_final['loss'] = loss / n_iters
    outputs_final['labels']= torch.stack(labels).numpy()
    return outputs_final


def calc_metrics_classification(
        outputs: Dict[str, Any],
        calc_distances: bool=False,
    ) -> Dict[str, float]:
    acc = calc_accuracy_np(outputs['preds'], outputs['labels'])
    ece = calc_ece_np(outputs['preds'], outputs['labels'], n_bins=15)
    ce, ent = calc_info_theory(outputs['preds'], outputs['labels'])
    metrics = {
        'Loss':     outputs['loss'], 
        'Top1':     acc, 
        'ECE':      ece,
        'CE':       ce,
        'Ent':      ent
    }
    if calc_distances:
        metrics.update(calc_feat_distance(outputs['features'], outputs['labels']))
    return metrics


def formatted_1_dict(
        dict_1: Dict[str, float],
        sep: str="\t| "
    ) -> str:
    return sep.join(
        "{}: {:.4f}".format(k, v) for k, v in dict_1.items()
    )

def formatted_2_dict(
        dict_2: Dict[str, Dict[str, float]],
        sep: str="\n"
    ) -> str:
    return sep.join(
        "{:6s}\t| {}".format(k, formatted_1_dict(v)) for k, v in dict_2.items()   
    )


def save_metrics(
        path: str, 
        metrics_classification: Dict[str, Dict[str, float]],
    ):
    with open(os.path.join(path, 'metrics_classification.yaml'), 'w') as outfile:
        yaml.dump(metrics_classification, outfile)


def save_scores(
        path: str, 
        scores_ood: Dict[str, Dict[str, np.ndarray]], 
        scores_test: Dict[str, np.ndarray],
    ):
    scores = {k: v for k, v in scores_ood.items()}
    scores['test'] = scores_test
    for dataset in scores:
        for postprocessor in scores[dataset]:
            scores[dataset][postprocessor] = scores[dataset][postprocessor].tolist()
    with open(os.path.join(path, 'scores.yaml'), 'w') as outfile:
        yaml.dump(scores, outfile)
 
def load_scores(path: str):
    with open(os.path.join(path, 'scores.yaml'), 'r') as file:
        scores = yaml.safe_load(file)
    return scores


def save_outpus(path, subset, outputs):
    np.savez(os.path.join(path, subset+'.npz'), **outputs)
    
def load_outputs(path, subset):
    outputs = np.load(os.path.join(path, subset+'.npz'))
    return {key: outputs[key] for key in outputs}