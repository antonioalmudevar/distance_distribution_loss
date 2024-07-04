"""
Modified version of:
https://github.com/pytorch/audio/blob/main/src/torchaudio/datasets/speechcommands.py
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torchaudio._internal import download_url_to_file

from src.datasets.base import BaseDataset
from src.datasets.utils import extract_tar, load_waveform, adjust_duration


FOLDER_IN_ARCHIVE = "SpeechCommands"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
SAMPLE_RATE = 16000
_CHECKSUMS = {
    "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz": "743935421bb51cccdb6bdd152e04c5c70274e935c82119ad7faeec31780d811d",  # noqa: E501
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",  # noqa: E501
}


def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output


def _get_speechcommands_metadata(filepath: str, path: str) -> Tuple[str, int, str, str, int]:
    """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.
        Args:
            filepath (str): path of the file
            path (str): path containing all the files
        Returns:
            Tuple of the following items;
            str: Path to the audio
            int: Sample rate
            str: Label
            str: Speaker ID
            int: Utterance number
        """
    relpath = os.path.relpath(filepath, path)
    reldir, filename = os.path.split(relpath)
    _, label = os.path.split(reldir)
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    return relpath, SAMPLE_RATE, label, speaker_id, utterance_number


class SpeechCommands(BaseDataset):

    n_classes = 35

    def __init__(
            self,
            root: Union[str, Path],
            version: int = 2,
            folder_in_archive: str = FOLDER_IN_ARCHIVE,
            download: bool = False,
            subset: Optional[str] = None,
            n_items: int = None,
            tensor_dir: str = None,
            **kwargs
        ) -> None:

        super().__init__(train=(subset=="training"), **kwargs)
        if subset is not None and subset not in ["training", "validation", "testing"]:
            raise ValueError("When `subset` is not None, it must be one of ['training', 'validation', 'testing'].")

        base_url = "http://download.tensorflow.org/data/"
        url = os.path.join(base_url, "speech_commands_v0.0{}.tar.gz".format(version))

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)

        self.tensor_path = None
        if tensor_dir is not None:
            tensor_dir = os.path.join(tensor_dir, FOLDER_IN_ARCHIVE, "data_tensors")
            if not os.path.exists(tensor_dir):
                os.makedirs(tensor_dir)
            self.tensor_path = os.path.join(tensor_dir, subset+".pt")
            

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                extract_tar(archive, self._path)
        else:
            if not os.path.exists(self._path):
                raise RuntimeError(
                    f"The path {self._path} doesn't exist. "
                    "Please check the ``root`` path or set `download=True` to download it"
                )

        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt")
        elif subset == "training":
            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]
        
        self.n_items = n_items
        self._load_metadata()
        self._load_targets()
        self._load_data()
        self.get_fbank_stats()
        #self._filter_by_class()


    def _load_metadata(self):
        self.metadata = []
        for i, fileid in enumerate(self._walker):
            self.metadata.append(_get_speechcommands_metadata(fileid, self._archive))


    def _load_targets(self):
        targets = [i[2] for i in self.metadata]
        self.targets = [list(set(targets)).index(i) for i in targets]


    def _load_data(self):
        if self.tensor_path is not None and os.path.exists(self.tensor_path):
            with open(self.tensor_path, 'rb') as f:
                data_tensor = torch.load(f)
            self.data = [row for row in data_tensor]
        else:
            for m in self.metadata:
                wav_original = load_waveform(self._archive, m[0], m[1])
                wav = adjust_duration(wav_original, 1*SAMPLE_RATE)
                self.data.append(wav)
            if self.tensor_path is not None:
                torch.save(torch.stack(self.data), self.tensor_path)


    def _filter_by_class(self):
        if self.n_items is None:
            pass
        else:
            cont_class = torch.zeros(self.n_classes)
            targets, data = [], []
            for i, target in enumerate(self.targets):
                if cont_class[target] < self.n_items:
                    targets.append(target)
                    data.append(self.data[i])
                    cont_class[target] += 1
                if cont_class.sum()==self.n_classes*self.n_items:
                    break
            self.targets = targets
            self.data = data
        