from typing import Tuple

from src.datasets import (
    BaseDataset, 
    ESC50, 
    IEMOCAP,
    SpeechCommands, 
    VoxCeleb1Identification,
)


def get_dataset(dataset: str, **kwargs) -> Tuple[BaseDataset, int]:
    if dataset.upper() in ["ESC50", "ESC-50"]:
        return get_esc50(**kwargs)
    elif dataset.upper() in ["SPEECHCOMMANDS", "SPEECH_COMMANDS"]:
        return get_speechcommands(**kwargs)
    elif dataset.upper() in ["IEMOCAP"]:
        return get_iemocap(**kwargs)
    elif dataset.upper() in ["IEMOCAP_ATTR"]:
        return get_iemocap_attr(**kwargs)
    elif dataset.upper() in ["VOXCELEB", "VOXCELEB1"]:
        return get_voxceleb1(**kwargs)
    else:
        raise ValueError
    

def get_esc50(
        root: str,
        **kwargs,
    ) -> Tuple[BaseDataset, int]:
    folds = [1,2,3,4,5]
    train_datasets = [ESC50(
        root=root, train=True, download=True, test_fold=fold, **kwargs
    ) for fold in folds]
    test_datasets = [ESC50(
        root=root, train=False, download=True, test_fold=fold, **kwargs
    ) for fold in folds]
    return train_datasets, test_datasets, 50


def get_speechcommands(
        root: str,
        **kwargs
    ) -> Tuple[BaseDataset, int]:

    train_datasets = [SpeechCommands(
        root=root, subset="training", download=True, **kwargs
    )]
    test_datasets = [SpeechCommands(
        root=root, subset="testing", download=True, **kwargs
    )]
    return train_datasets, test_datasets, train_datasets[0].n_classes


def get_iemocap(
        root: str,
        **kwargs,
    ) -> Tuple[BaseDataset, int]:
    folds = [1,2,3,4,5]
    train_datasets = [IEMOCAP(
        root=root, train=True, sessions=[i for i in folds if i!=fold], **kwargs
    ) for fold in folds]
    test_datasets = [IEMOCAP(
        root=root, train=True, sessions=[fold], **kwargs
    ) for fold in folds]
    return train_datasets, test_datasets, train_datasets[0].n_classes


def get_iemocap_attr(
        root: str,
        **kwargs,
    ) -> Tuple[BaseDataset, int]:
    folds = [1,2,3,4,5]
    train_datasets = [IEMOCAP(
        root=root, train=True, sessions=[i for i in folds if i!=fold], **kwargs
    ) for fold in folds]
    test_datasets = [IEMOCAP(
        root=root, train=True, sessions=[fold], **kwargs
    ) for fold in folds]
    return train_datasets, test_datasets, train_datasets[0].n_classes



def get_voxceleb1(
        root: str,
        train: bool=True,
        **kwargs
    ) -> Tuple[BaseDataset, int]:

    dataset = VoxCeleb1Identification(
        root=root, 
        subset="train" if train else "test", 
        download=True,
        **kwargs
    )
    
    return dataset, dataset.n_classes