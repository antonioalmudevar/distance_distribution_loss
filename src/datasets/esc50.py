"""
Modified version of:
https://github.com/AminJun/torchaudio.dataset.ESC-50/blob/master/ESC-50.py
"""
import urllib.request
import zipfile
import os

from torchvision.datasets.utils import check_integrity
import pandas as pd
import torchaudio

from src.datasets.base import BaseDataset


class ESC50(BaseDataset):
    base_folder = 'ESC-50-master'
    url = "https://codeload.github.com/karolpiczak/ESC-50/zip/master"
    filename = "ESC-50-master.zip"
    zip_md5 = '70cce0ef1196d802ae62ce40db11b620'
    num_files_in_dir = 2000
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': 'meta/esc50.csv',
        'md5': '54a0d0055a10bb7df84ad340a148722e',
    }
    n_classes = 50

    def __init__(
            self, 
            root: str, 
            download: bool = True,
            train: bool=True, 
            test_fold: int=5,
            **kwargs
        ):
        super().__init__(train=train, **kwargs)
        self.root = os.path.expanduser(root)
        if download:
            self.download()
        self.train = train
        self.test_fold = test_fold
        self._load_meta()
        self._load_data()
        self.get_fbank_stats()
        

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')

        data = pd.read_csv(path)
        index = data['fold'] != self.test_fold if self.train else data['fold'] == self.test_fold
        self.df = data[index]
        self.class_to_idx = {}
        self.classes = sorted(self.df[self.label_col].unique())
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i


    def _load_data(self):
        for _, row in self.df.iterrows():
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            wav, sr = torchaudio.load(file_path)
            if sr!=self.sample_frequency:
                resampler = torchaudio.transforms.Resample(sr, self.sample_frequency)
                wav = resampler(wav)
            self.data.append(wav)
            self.targets.append(self.class_to_idx[row[self.label_col]])


    def download(self):
        if not os.path.exists(os.path.join(self.root, self.base_folder)):
            zip_filename = os.path.join(self.root, self.filename)
            urllib.request.urlretrieve(self.url, zip_filename)
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            os.remove(zip_filename)
            print(f"File downloaded and extracted to {self.root}")
        else:
            print(f"File '{self.filename}' already exists in {self.root}")