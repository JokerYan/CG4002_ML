import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


class HaptRawDataset(Dataset):
    def __init__(self, data_json_path, transform=None):
        self.transform = transform
        with open(data_json_path, 'rb') as data_json_file:
            self.data_json = pickle.load(data_json_file)

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        acc_window = np.asarray(self.data_json[idx]['acc_window'], dtype=np.float32).transpose()  # C x L
        gyro_window = np.asarray(self.data_json[idx]['gyro_window'], dtype=np.float32).transpose()  # C x L
        data_window = np.vstack([acc_window, gyro_window])
        label = np.asarray(self.data_json[idx]['label'], dtype=np.int64)
        label = label - 1
        item = data_window, label
        if self.transform:
            item = self.transform(item)
        return item