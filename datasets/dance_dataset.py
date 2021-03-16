import torch
from torch.utils.data import Dataset


class DanceDataset(Dataset):
    def __init__(self, feature_data_list, transform=None):
        self.feature_data_list = feature_data_list
        self.transform = transform

    def __len__(self):
        return len(self.feature_data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = self.feature_data_list[idx]
        if self.transform:
            item = self.transform(item)
        return item

    def input_size(self):
        return self.feature_data_list[0][0].shape[0]