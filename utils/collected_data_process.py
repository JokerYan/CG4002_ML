import os
import re
import pickle
from glob import glob
import numpy as np

from utils.feature_extraction import extract_instance_feature

data_root = r'C:\Code\Projects\Dance\Data'
data_folder = r'data_collect_v1'

data_rate = 20  # assume 20Hz
window_size = 30  # 1.5s
discarded_size = 600  # 30s discarded at the start

normalize_output_min = -1
normalize_output_max = 1

dance_move_idx_dict = {
    'still': 0,
    'gun': 1,
    'hair': 2,
    'sidepump': 3,
}


def get_window_from_raw(raw_data):
    data = raw_data[:, discarded_size:]
    data_length = data.shape[1]
    window_count = data_length // (window_size // 2) - 1
    windows = []
    for i in range(0, window_count):
        window_start = i * window_size // 2
        window_end = i * window_size // 2 + window_size
        window_data = data[:, window_start:window_end]
        windows.append(window_data)
    return windows


def get_window_raw_data_list():
    window_raw_data_list = []
    for file_path in glob(os.path.join(data_root, data_folder, '*')):
        match = re.search(r'\\([^\\]*)_([^\\]*)_(?:[^\\]*)$', file_path)
        dancer_id = match.group(1)
        dance_move = match.group(2)
        label = dance_move_idx_dict[dance_move]

        data = pickle.load(open(file_path, 'rb'))
        windows = get_window_from_raw(data)
        for window in windows:
            window_raw_data_list.append((window, label))
    return window_raw_data_list


def get_feature_data_list(window_raw_data_list):
    feature_data_list = []
    for window_raw_data in window_raw_data_list:
        feature_vec, label = extract_instance_feature(window_raw_data)
        feature_vec = np.array(feature_vec, dtype=np.float32)
        feature_data_list.append((feature_vec, label))
    return feature_data_list


# using simple min max
def normalize_feature_data_list(feature_data_list):
    feature_matrix = [feature_data for feature_data, label in feature_data_list]
    feature_matrix = np.array(feature_matrix).transpose()
    feature_max = np.max(feature_matrix, axis=1).reshape((-1, 1))
    feature_min = np.min(feature_matrix, axis=1).reshape((-1, 1))
    feature_range = (feature_max - feature_min) / (normalize_output_max - normalize_output_min)
    # feature_normalized_matrix = np.subtract(feature_matrix, feature_min)
    feature_normalized_matrix = (feature_matrix - feature_min) / feature_range + normalize_output_min
    for i in range(len(feature_data_list)):
        label = feature_data_list[i][1]
        feature_data_list[i] = feature_normalized_matrix[:, i], label
    return feature_data_list


def get_collected_data():
    window_raw_data_list = get_window_raw_data_list()
    feature_data_list = get_feature_data_list(window_raw_data_list)
    normalized_feature_data_list = normalize_feature_data_list(feature_data_list)
    return normalized_feature_data_list


def main():
    normalized_feature_data_list = get_collected_data()
    pickle.dump(normalized_feature_data_list, open('collected_data_processed.pickle', 'wb'))


if __name__ == '__main__':
    main()