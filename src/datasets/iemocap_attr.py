import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List
import json
import statistics as stats
import numpy as np
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio


def _get_wavs_paths(data_dir):
    wav_dir = data_dir / "sentences" / "wav"
    wav_paths = sorted(str(p) for p in wav_dir.glob("*/*.wav"))
    relative_paths = []
    for wav_path in wav_paths:
        start = wav_path.find("Session")
        wav_path = wav_path[start:]
        relative_paths.append(wav_path)
    return relative_paths


def remove_elements_over_threshold(tensor, threshold):
    mask = tensor <= threshold  # Create a mask of True/False values based on the condition
    filtered_tensor = tensor[mask]  # Apply the mask to the tensor
    return filtered_tensor


def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
    

class IEMOCAPAttr(Dataset):

    DEFAULT_ATTRIBUTES = {
        'pitch_median':     [85, 275],
        'pitch_std':        [0, 10000],
        'loudness':         [-50, 0],
    }

    DEFAULT_N_PERCENTILES = {
        'pitch_median':     6,
        'pitch_std':        6,
        'loudness':         6,
    }

    LABELS = ["neu", "hap", "ang", "sad", "exc", "fru"]

    def __init__(
        self,
        root: Union[str, Path],
        sessions: Tuple[str] = (1, 2, 3, 4, 5),
        utterance_type: Optional[str] = None,
        labels: List[str]= LABELS,
        attributes: Dict = DEFAULT_ATTRIBUTES,
        gender: List = ['F', 'M'],
        n_percentiles = None,
        length: int = 41200,
    ):
        root = Path(root)
        self._path = root

        labels = [l for l in labels if l!="exc"]
        self.labels = labels

        self.gender = gender
        n_percentiles = self.DEFAULT_N_PERCENTILES if n_percentiles is None else n_percentiles
        self.length = length

        self.dict_path = Path(__file__).resolve().parents[2] / "data" /"IEMOCAP"
        Path(self.dict_path).mkdir(parents=True, exist_ok=True)

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found.")

        if utterance_type not in ["scripted", "improvised", None]:
            raise ValueError("utterance_type must be one of ['scripted', 'improvised', or None]")
        
        self._load_data(sessions, utterance_type, attributes)
        self._percentilaze(n_percentiles)


    def _load_data(self, sessions, utterance_type, attributes):

        sessions = [sessions] if isinstance(sessions, int) else sessions
        data_fn = 'data_' + '_'.join(str(element) for element in sessions) + '.json'

        if (self.dict_path / data_fn).exists():

            with open(self.dict_path / data_fn, 'r') as json_file:
                self.mapping = json.load(json_file)

            self.data = [wav_stem for wav_stem in self.mapping]

        else:

            all_data = []
            self.data = []
            self.mapping = {}

            for session in sessions:
                session_name = f"Session{session}"
                session_dir = self._path / session_name

                # get wav paths
                wav_paths = _get_wavs_paths(session_dir)
                for wav_path in wav_paths:
                    wav_stem = str(Path(wav_path).stem)
                    if wav_stem.split("_")[-1][0] in self.gender:
                        all_data.append(wav_stem)

                # add labels
                label_dir = session_dir / "dialog" / "EmoEvaluation"
                query = "*.txt"
                if utterance_type == "scripted":
                    query = "*script*.txt"
                elif utterance_type == "improvised":
                    query = "*impro*.txt"
                label_paths = label_dir.glob(query)

                for label_path in label_paths:
                    with open(label_path, "r") as f:
                        for line in f:
                            if not line.startswith("["):
                                continue
                            line = re.split("[\t\n]", line)
                            wav_stem = line[1]
                            label = line[2]
                            if wav_stem not in all_data:
                                continue
                            if label=="exc":
                                label = "hap"
                            if label not in self.labels:
                                continue
                            self.mapping[wav_stem] = {}
                            self.mapping[wav_stem]["label"] = label

                # add wavs
                for wav_path in wav_paths:
                    wav_stem = str(Path(wav_path).stem)
                    if wav_stem in self.mapping:
                        wav, sr = torchaudio.load(self._path/wav_path)
                        pitch = torchaudio.functional.detect_pitch_frequency(wav, sr)
                        pitch = remove_elements_over_threshold(pitch, attributes['pitch_median'][1])
                        loudness = torchaudio.functional.loudness(wav, sr)
                        self.mapping[wav_stem]["pitch_median"] = pitch.median()
                        self.mapping[wav_stem]["pitch_std"] = pitch.std()
                        self.mapping[wav_stem]["loudness"] = loudness
                        self.mapping[wav_stem]["wav"] = wav[0]

            mapping_copy = self.mapping.copy()
            for wav_stem in mapping_copy:
                if torch.isnan(mapping_copy[wav_stem]['pitch_median']):
                    self.mapping.pop(wav_stem)
                elif torch.isnan(mapping_copy[wav_stem]['pitch_std']):
                    self.mapping.pop(wav_stem)
                elif mapping_copy[wav_stem]['pitch_median']<attributes['pitch_median'][0]:
                    self.mapping.pop(wav_stem)
                elif mapping_copy[wav_stem]['pitch_median']>attributes['pitch_median'][1]:
                    self.mapping.pop(wav_stem)
                elif mapping_copy[wav_stem]['loudness']<attributes['loudness'][0]:
                    self.mapping.pop(wav_stem)
                elif mapping_copy[wav_stem]['loudness']>attributes['loudness'][1]:
                    self.mapping.pop(wav_stem)
                elif mapping_copy[wav_stem]['wav'].shape[0] < self.length:
                    self.mapping.pop(wav_stem)
                else:
                    self.data.append(wav_stem)

            """with open(self.dict_path/data_fn, 'w') as json_file:
                json.dump(convert_to_serializable(self.mapping), json_file)
"""

    def _find_percentiles(self, n_percentiles):
        self.percentiles = {feature : stats.quantiles(
            [self.mapping[wav_stem][feature] for wav_stem in self.mapping], 
            n=n_percentiles[feature]
        ) for feature in n_percentiles}


    def _between_what_percentiles(self, value, percentiles):
        for i, percentile in enumerate(percentiles):
            if value < percentile:
                return i
        return i+1
        

    def _percentilaze(self, n_percentiles):
        self._find_percentiles(n_percentiles)
        for wav_stem in self.mapping:
            for feature in self.percentiles:
                self.mapping[wav_stem][feature] = self._between_what_percentiles(
                    self.mapping[wav_stem][feature], self.percentiles[feature]
                )
        print(self.percentiles)


    def __getitem__(self, n: int) -> Tuple[Tensor, str, Dict]:

        wav_stem = self.data[n]
        wav = self.mapping[wav_stem]["wav"]
        start_frame = np.int64(random.random()*(wav.shape[0]-self.length))
        wav = wav[start_frame:start_frame + self.length]
        label = torch.tensor(self.labels.index(self.mapping[wav_stem]["label"]))
        speaker = wav_stem.split("_")[0]
        attributes = {
            feature: self.mapping[wav_stem][feature] for feature in self.percentiles
        }
        return (wav, label, attributes)


    def __len__(self):
        return len(self.mapping)