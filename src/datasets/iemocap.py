"""
Modified version of:
https://github.com/pytorch/audio/blob/main/src/torchaudio/datasets/iemocap.py
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union, List

from src.datasets.base import BaseDataset
from src.datasets.utils import load_waveform, adjust_duration


_SAMPLE_RATE = 16000


def _get_wavs_paths(data_dir):
    wav_dir = data_dir / "sentences" / "wav"
    wav_paths = sorted(str(p) for p in wav_dir.glob("*/*.wav"))
    relative_paths = []
    for wav_path in wav_paths:
        start = wav_path.find("Session")
        wav_path = wav_path[start:]
        relative_paths.append(wav_path)
    return relative_paths


class IEMOCAP(BaseDataset):
    """*IEMOCAP* :cite:`iemocap` dataset.

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found
        sessions (Tuple[int]): Tuple of sessions (1-5) to use. (Default: ``(1, 2, 3, 4, 5)``)
        utterance_type (str or None, optional): Which type(s) of utterances to include in the dataset.
            Options: ("scripted", "improvised", ``None``). If ``None``, both scripted and improvised
            data are used.
    """

    def __init__(
            self,
            root: Union[str, Path],
            train: bool = True,
            sessions: Tuple[str] = (1, 2, 3, 4, 5),
            labels: List[str] = ["neu", "hap", "ang", "sad", "exc", "fru"],
            utterance_type: Optional[str] = None,
            length: int = 41200,
            **kwargs
        ):
        super().__init__(train=train, **kwargs)
        
        root = Path(root)
        self._path = root / "IEMOCAP"
        self.length = length

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found.")

        if utterance_type not in ["scripted", "improvised", None]:
            raise ValueError("utterance_type must be one of ['scripted', 'improvised', or None]")

        all_data = []
        self.data_stem = []
        self.mapping = {}

        for session in sessions:
            session_name = f"Session{session}"
            session_dir = self._path / session_name

            # get wav paths
            wav_paths = _get_wavs_paths(session_dir)
            for wav_path in wav_paths:
                wav_stem = str(Path(wav_path).stem)
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
                        if label not in labels:
                            continue
                        self.mapping[wav_stem] = {}
                        self.mapping[wav_stem]["label"] = label

            for wav_path in wav_paths:
                wav_stem = str(Path(wav_path).stem)
                if wav_stem in self.mapping:
                    self.data_stem.append(wav_stem)
                    self.mapping[wav_stem]["path"] = wav_path

        self._load_metadata()
        self._load_targets()
        self._load_data()
        self.get_fbank_stats()


    def _load_metadata(self):
        self.metadata = []
        for wav_stem in self.data_stem:
            wav_path = self.mapping[wav_stem]["path"]
            label = self.mapping[wav_stem]["label"]
            speaker = wav_stem.split("_")[0]
            self.metadata.append((wav_path, _SAMPLE_RATE, wav_stem, label, speaker))


    def _load_targets(self):
        targets = [i[3] for i in self.metadata]
        self.n_classes = len(set(targets))
        self.targets = [list(set(targets)).index(i) for i in targets]


    def _load_data(self):
        self.data = [adjust_duration(
            load_waveform(self._path, m[0], m[1]), self.length
        ) for m in self.metadata]