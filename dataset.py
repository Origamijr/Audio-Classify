import os
from scipy.ndimage.measurements import label
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

from config import CONFIG
from preprocessing import get_directory

class H5SpecSeqDataset(Dataset):
    """
    Dataset for sequences of 2D features from an HDF5 file
    """

    def __init__(self, hdf_file=CONFIG['preprocessing']['destination'], transform=None):
        label_key = CONFIG['preprocessing']['hdf_label_key']

        self.labels = pd.read_hdf(hdf_file, key=label_key)
        
        dfs = []
        for label in tqdm(self.labels, desc='Reading File', smoothing=0.1):
            dfs += [pd.read_hdf(hdf_file, key=label)]
        self.df = pd.concat(dfs, ignore_index=True)

        # TODO Data normalization?
        self.transform = transform

    
    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        spec = row['magnitude']
        label = get_directory(row['file'])

        if self.transform is not None:
            spec = self.transform(spec)

        return spec, self.get_label_index(label)


    def get_label_index(self, label):
        return self.labels[self.labels == label].index[0]

    
    def get_label(self, idx):
        return self.labels[idx]
