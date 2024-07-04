"""
https://github.com/pytorch/audio/blob/main/src/torchaudio/datasets/utils.py
"""

import logging
import os
import tarfile
import zipfile
from typing import List, Optional

import torch
import torchaudio

_LG = logging.getLogger(__name__)


def extract_tar(
        from_path: str, 
        to_path: Optional[str] = None, 
        overwrite: bool = False
    ) -> List[str]:
    if to_path is None:
        to_path = os.path.dirname(from_path)
    with tarfile.open(from_path, "r") as tar:
        files = []
        for file_ in tar:  # type: Any
            file_path = os.path.join(to_path, file_.name)
            if file_.isfile():
                files.append(file_path)
                if os.path.exists(file_path):
                    _LG.info("%s already extracted.", file_path)
                    if not overwrite:
                        continue
            tar.extract(file_, to_path)
        return files


def extract_zip(
        from_path: str, 
        to_path: Optional[str] = None, 
        overwrite: bool = False
    ) -> List[str]:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as zfile:
        files = zfile.namelist()
        for file_ in files:
            file_path = os.path.join(to_path, file_)
            if os.path.exists(file_path):
                _LG.info("%s already extracted.", file_path)
                if not overwrite:
                    continue
            zfile.extract(file_, to_path)
    return files


def load_waveform(
        root: str,
        filename: str,
        exp_sample_rate: int,
    ) -> torch.Tensor:
    path = os.path.join(root, filename)
    waveform, sample_rate = torchaudio.load(path)
    if exp_sample_rate != sample_rate:
        raise ValueError(f"sample rate should be {exp_sample_rate}, but got {sample_rate}")
    return waveform


def adjust_duration(
        wav: torch.Tensor,
        target_duration: int
    ):
    if target_duration==wav.shape[1]:
        return wav
    elif target_duration>wav.shape[1]:
        padded_wav = torch.zeros(1, target_duration).to(dtype=wav.dtype)
        padded_wav[0,:wav.shape[1]] = wav
        return padded_wav
    else:
        return wav[:,:target_duration]