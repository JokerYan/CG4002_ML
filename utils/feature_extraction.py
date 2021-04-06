import os
import numpy as np
from scipy.fft import fft, fftfreq

from datasets.hapt_raw_dataset import HaptRawDataset
from sklearn.linear_model import LinearRegression

data_root = r"C:\Code\Projects\Dance\Data\HAPT Data Set"
output_pickle_path = os.path.join(data_root, 'raw_window_data.pickle')

train_x_data_path = os.path.join(data_root, r"Train\X_train_extracted.txt")
train_y_data_path = os.path.join(data_root, r"Train\y_train_extracted.txt")


def extract_mean(raw_data):
    mean_vec = np.mean(raw_data, axis=1)
    return mean_vec


def extract_std(raw_data):
    std_vec = np.std(raw_data, axis=1)
    return std_vec


def extract_max(raw_data):
    max_vec = np.max(raw_data, axis=1)
    return max_vec


def extract_min(raw_data):
    min_vec = np.min(raw_data, axis=1)
    return min_vec


def extract_linear(raw_data):
    x = np.array([i for i in range(raw_data.shape[1])]).reshape(-1, 1)
    coef_list = []
    for data in raw_data:
        reg = LinearRegression().fit(x, data)
        coef_list.append(reg.coef_[0])
    return np.array(coef_list)


def extract_fft(data):
    yf_list = []
    for sensor_data in data:
        N = 20
        T = 1 / 20
        yf = fft(sensor_data)
        yf = 2.0 / N * np.abs(yf[0:N // 2])
        yf_list.append(yf)
    return np.array(yf_list).reshape(-1)


def extract_instance_feature(data_instance):
    raw_data, label = data_instance
    feature_extractions = [
        extract_mean,
        extract_std,
        extract_max,
        extract_min,
        extract_linear,
        extract_fft,
    ]
    feature_vec = np.array([])
    for extraction in feature_extractions:
        vec = extraction(raw_data)
        feature_vec = np.hstack([feature_vec, vec])
    return feature_vec, label


def extract_raw_dataset_feature(raw_dataset):
    feature_dataset = []
    for i in range(len(raw_dataset)):
        feature_vec, label = extract_instance_feature(raw_dataset[i])
        feature_dataset.append((feature_vec, label))
    return feature_dataset


def save_feature_dataset(feature_dataset):
    feature_str_list = []
    label_str_list = []
    for data in feature_dataset:
        feature, label = data
        feature_str = " ".join([str(f) for f in feature])
        label_str = str(label)
        feature_str_list.append(feature_str)
        label_str_list.append(label_str)
    with open(train_x_data_path, 'w+') as train_x_file:
        train_x_file.write('\n'.join(feature_str_list))
    with open(train_y_data_path, 'w+') as train_y_file:
        train_y_file.write('\n'.join(label_str_list))


def main():
    dataset = HaptRawDataset(output_pickle_path)
    # extract_instance_feature(dataset.__getitem__(0))
    feature_dataset = extract_raw_dataset_feature(dataset)
    save_feature_dataset(feature_dataset)


if __name__ == '__main__':
    main()