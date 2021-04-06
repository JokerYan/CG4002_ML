import os
import re
import pickle
from glob import glob
import numpy as np

from utils.feature_extraction import extract_instance_feature

data_root = r'C:\Code\Projects\Dance\CG4002\Ultra96'
data_folder = r'data'
val_data_folder = r'val_data'

data_rate = 20  # assume 20Hz
window_size = 50  # 2.5s
discarded_size = 600  # 30s discarded at the start
reading_count = 12  # two bluno, each 6 readings

normalize_output_min = -1
normalize_output_max = 1

dance_move_idx_dict = {
    'still': 0,
    'gun': 1,
    'hair': 2,
    'sidepump': 3,
    'pointhigh': 4,
    'elbowkick': 5,
    'listen': 6,
    'dab': 7,
    'wipetable': 8,
    'logout': 9,
    'left': 10,
    'right': 11,
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


def get_window_raw_data_list(val=False):
    window_raw_data_list = []
    target_data_folder = data_folder if not val else val_data_folder
    for file_path in glob(os.path.join(data_root, target_data_folder, '*')):
        # if 'left' not in file_path and 'right' not in file_path and 'still' not in file_path:
        #     continue
        # if 'left' in file_path or 'right' in file_path:
        #     continue
        match = re.search(r'\\([^\\]*)_([^\\]*)_(?:[^\\]*)$', file_path)
        dancer_id = match.group(1)
        dance_move = match.group(2)
        label = dance_move_idx_dict[dance_move]

        data = pickle.load(open(file_path, 'rb'))
        data = data[:reading_count][:]
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


def save_normalize_data(feature_min, feature_max):
    pickle.dump({
        "feature_min": feature_min,
        "feature_max": feature_max,
    }, open('feature_normalize_data.pickle', 'wb'))

def load_normalize_data():
    features = pickle.load(open('feature_normalize_data.pickle', 'rb'))
    feature_min = features['feature_min'].reshape(-1)
    feature_max = features['feature_max'].reshape(-1)
    return feature_min, feature_max

# using simple min max
def normalize_feature_data_list(feature_data_list, val=False):
    feature_matrix = [feature_data for feature_data, label in feature_data_list]
    feature_matrix = np.array(feature_matrix).transpose()
    feature_max = np.max(feature_matrix, axis=1).reshape((-1, 1))
    feature_min = np.min(feature_matrix, axis=1).reshape((-1, 1))
    if val:
        feature_min, feature_max = load_normalize_data()
        feature_min, feature_max = feature_min.reshape((-1, 1)), feature_max.reshape((-1, 1))
    else:
        save_normalize_data(feature_min, feature_max)
    feature_range = (feature_max - feature_min) / (normalize_output_max - normalize_output_min)
    feature_normalized_matrix = (feature_matrix - feature_min) / feature_range + normalize_output_min
    for i in range(len(feature_data_list)):
        label = feature_data_list[i][1]
        feature_data_list[i] = feature_normalized_matrix[:, i], label
    return feature_data_list


def get_collected_data(val=False):
    window_raw_data_list = get_window_raw_data_list(val)
    feature_data_list = get_feature_data_list(window_raw_data_list)
    normalized_feature_data_list = normalize_feature_data_list(feature_data_list, val)
    return normalized_feature_data_list


def main():
    normalized_feature_data_list = get_collected_data()
    pickle.dump(normalized_feature_data_list, open('collected_data_processed.pickle', 'wb'))


if __name__ == '__main__':
    main()