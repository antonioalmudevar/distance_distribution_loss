import os
import random

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking
import torchaudio.compliance.kaldi as ta_kaldi


class BaseDataset(Dataset):

    n_classes = 10

    def __init__(
            self, 
            stats_dir: str,
            train: bool = True,
            num_mel_bins: int = 128, 
            sample_frequency: int = 16000,
            frame_length: int = 25, 
            frame_shift: int = 10,
            htk_compat: bool = False,
            use_energy: bool = False,
            window_type: str = "hanning",
            freq_mask_param: int = 0,
            time_mask_param: int = 0,
            mixup: float = 0.,
            fbank_mean: float = None,
            fbank_std: float = None,
        ):
        super().__init__()
        self.stats_dir = stats_dir
        self.train = train
        self.num_mel_bins = num_mel_bins
        self.sample_frequency = sample_frequency
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.htk_compat = htk_compat
        self.use_energy = use_energy
        self.window_type = window_type
        self.freqm = FrequencyMasking(freq_mask_param) if freq_mask_param>0 and train else nn.Identity()
        self.timem = TimeMasking(time_mask_param) if time_mask_param>0 and train else nn.Identity()
        self.mixup = mixup
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std
        self.data, self.targets = [], []


    def wav2fbank(self, wav: Tensor, mask: bool=True, norm: bool=True) -> Tensor:
        #wav = wav * 2 ** 15
        wav = wav - wav.mean()
        fbank = ta_kaldi.fbank(
            waveform=wav, 
            num_mel_bins=self.num_mel_bins, 
            sample_frequency=self.sample_frequency, 
            frame_length=self.frame_length, 
            frame_shift=self.frame_shift,
            htk_compat=self.htk_compat,
            use_energy=self.use_energy,
            window_type=self.window_type
        )
        if mask and self.train:
            fbank = torch.transpose(fbank, 0, 1)
            fbank = fbank.unsqueeze(0)
            fbank = self.timem(self.freqm(fbank))
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)
        if norm:
            fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)
        return fbank
        
    
    def get_fbank_stats(self):
        stats_path = os.path.join(self.stats_dir, "fbank_stats.npz")
        if os.path.exists(stats_path):
            loaded_arrays = np.load(stats_path)
            fbank_mean, fbank_std = loaded_arrays['mean'], loaded_arrays['std']
        else:
            assert self.train, "Mean and Std must be calculated for training dataset"
            fbanks = torch.stack([self.wav2fbank(wav, mask=False, norm=False) for wav in self.data])
            fbank_mean, fbank_std = fbanks.mean(), fbanks.std()
            np.savez(stats_path, mean=fbank_mean, std=fbank_std)
        if self.fbank_mean is None:
            self.fbank_mean = fbank_mean
        if self.fbank_std is None:
            self.fbank_std = fbank_std


    def get_fbank_shape(self):
        return self.wav2fbank(self.data[0]).shape


    def __getitem__(self, idx: int):
        wav = self.data[idx]
        target = self.targets[idx]
        target = 1. * F.one_hot(torch.tensor(self.targets[idx]), self.n_classes)
        if random.random() < self.mixup and self.train:
            idx_mix = random.randint(0, len(self.data)-1)
            wav_mix = self.data[idx_mix]
            target_mix = F.one_hot(torch.tensor(self.targets[idx_mix]), self.n_classes)
            lambda_mix = np.random.beta(10, 10)
            wav = lambda_mix * wav + (1 - lambda_mix) * wav_mix
            target = lambda_mix * target + (1 - lambda_mix) * target_mix
        return self.wav2fbank(wav), target


    def __len__(self):
        return len(self.data)