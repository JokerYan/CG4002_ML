import re

import torch
import numpy as np
from torch.utils.data import Dataset

def array_to_one_hot(array, max_class=None):
    array = np.array(array)
    if not max_class:
        max_class = array.max()
    one_hot = np.zeros((array.size, max_class))
    one_hot[np.arange(array.size), array] = 1
    return one_hot

class HaptDataset(Dataset):
    def __init__(self, x_file_path, y_file_path, transform=None, target_features=None):
        self.transform = transform
        with open(x_file_path, 'r') as x_file:
            lines = x_file.readlines()
            data_str_list = [line.strip().split(" ") for line in lines]
            self.input_list = np.array([[float(x) for x in data_point] for data_point in data_str_list])

        if target_features:
            self.select_feature(target_features)

        with open(y_file_path, 'r') as y_file:
            lines = y_file.readlines()
            label_list = [[int(line.strip()) - 1] for line in lines]  # Important, label becomes 0 indexing here
            # self.target_list = array_to_one_hot(label_list)
            self.target_list = np.array(label_list)
        assert len(self.input_list) == len(self.target_list)

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = torch.from_numpy(self.input_list[idx]).float(), \
               torch.from_numpy(self.target_list[idx]).long()
        if self.transform:
            item = self.transform(item)
        return item

    def get_mean_std(self, indices):
        subset = self.input_list[indices].transpose()
        return np.mean(subset, axis=1), np.std(subset, axis=1)

    def normalize(self, mean, std):
        self.input_list = (self.input_list - mean) / std

    def select_feature(self, target_features):
        with open('features.txt', 'r') as feature_list_file:
            feature_list = [line.strip() for line in feature_list_file.readlines()]
        feature_idx_list = []
        for i, feature in enumerate(feature_list):
            for target_feature in target_features:
                if re.search(target_feature, feature):
                    feature_idx_list.append(i)
                    break
        self.input_list = self.input_list[:, feature_idx_list]
        print('Number of features selected: {}'.format(self.input_list.shape[1]))
        return self.input_list

    def input_size(self):
        return self.input_list.shape[1]