"""
Modified version of:
https://github.com/pytorch/audio/blob/main/src/torchaudio/datasets/voxceleb1.py
"""

import os
from pathlib import Path
from typing import List, Union

from torchaudio._internal import download_url_to_file

from src.datasets.base import BaseDataset
from src.datasets.utils import extract_zip, load_waveform


SAMPLE_RATE = 16000
_ARCHIVE_CONFIGS = {
    "dev": {
        "archive_name": "vox1_dev_wav.zip",
        "urls": [
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa",
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab",
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac",
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad",
        ],
        "checksums": [
            "21ec6ca843659ebc2fdbe04b530baa4f191ad4b0971912672d92c158f32226a0",
            "311d21e0c8cbf33573a4fce6c80e5a279d80736274b381c394319fc557159a04",
            "92b64465f2b2a3dc0e4196ae8dd6828cbe9ddd1f089419a11e4cbfe2e1750df0",
            "00e6190c770b27f27d2a3dd26ee15596b17066b715ac111906861a7d09a211a5",
        ],
    },
    "test": {
        "archive_name": "vox1_test_wav.zip",
        "url": "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip",
        "checksum": "8de57f347fe22b2c24526e9f444f689ecf5096fc2a92018cf420ff6b5b15eaea",
    },
}
_IDEN_SPLIT_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"


def _download_extract_wavs(root: str):
    for archive in ["dev", "test"]:
        archive_name = _ARCHIVE_CONFIGS[archive]["archive_name"]
        archive_path = os.path.join(root, archive_name)
        # The zip file of dev data is splited to 4 chunks.
        # Download and combine them into one file before extraction.
        if archive == "dev":
            urls = _ARCHIVE_CONFIGS[archive]["urls"]
            checksums = _ARCHIVE_CONFIGS[archive]["checksums"]
            with open(archive_path, "wb") as f:
                for url, checksum in zip(urls, checksums):
                    file_path = os.path.join(root, os.path.basename(url))
                    download_url_to_file(url, file_path, hash_prefix=checksum)
                    with open(file_path, "rb") as f_split:
                        f.write(f_split.read())
        else:
            url = _ARCHIVE_CONFIGS[archive]["url"]
            checksum = _ARCHIVE_CONFIGS[archive]["checksum"]
            download_url_to_file(url, archive_path, hash_prefix=checksum)
        extract_zip(archive_path)


def _get_flist(root: str, file_path: str, subset: str) -> List[str]:
    f_list = []
    if subset == "train":
        index = 1
    elif subset == "dev":
        index = 2
    else:
        index = 3
    with open(file_path, "r") as f:
        for line in f:
            id, path = line.split()
            if int(id) == index:
                f_list.append(path)
    return sorted(f_list)


def _get_file_id(file_path: str, _ext_audio: str):
    speaker_id, youtube_id, utterance_id = file_path.split("/")[-3:]
    utterance_id = utterance_id.replace(_ext_audio, "")
    file_id = "-".join([speaker_id, youtube_id, utterance_id])
    return file_id



class VoxCeleb1Identification(BaseDataset):

    _ext_audio = ".wav"

    def __init__(
            self, 
            root: Union[str, Path], 
            subset: str = "train", 
            meta_url: str = _IDEN_SPLIT_URL, 
            download: bool = False,
            **kwargs
        ) -> None:
        super().__init__(train=(subset=="train"), **kwargs)

        root = os.fspath(root)
        self._path = os.path.join(root, "wav")
        if not os.path.isdir(self._path):
            if not download:
                raise RuntimeError(
                    f"Dataset not found at {self._path}. Please set `download=True` to download the dataset."
                )
            _download_extract_wavs(root)

        if subset not in ["train", "dev", "test"]:
            raise ValueError("`subset` must be one of ['train', 'dev', 'test']")
        # download the iden_split.txt to get the train, dev, test lists.
        meta_list_path = os.path.join(root, os.path.basename(meta_url))
        if not os.path.exists(meta_list_path):
            download_url_to_file(meta_url, meta_list_path)
        self._flist = _get_flist(self._path, meta_list_path, subset)


    def _load_metadata(self):
        self.metadata = []
        for n in len(self._flist):
            file_path = self._flist[n]
            file_id = _get_file_id(file_path, self._ext_audio)
            speaker_id = file_id.split("-")[0]
            speaker_id = int(speaker_id[3:])
            self.metadata.append((file_path, SAMPLE_RATE, speaker_id, file_id))


    def _load_targets(self):
        targets = [i[2] for i in self.metadata]
        self.n_classes = len(set(targets))
        self.targets = [list(set(targets)).index(i) for i in targets]


    def _load_data(self):
        self.data = [load_waveform(self._flist, m[0], m[1]) for m in self.metadata]