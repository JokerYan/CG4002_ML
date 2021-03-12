import os
import pickle

from tqdm import tqdm

from datasets.hapt_raw_dataset import HaptRawDataset

data_root = r"C:\Code\Projects\Dance\Data\HAPT Data Set"
raw_dir = os.path.join(data_root, "RawData")
label_dir = os.path.join(raw_dir, 'labels.txt')
output_pickle_path = os.path.join(data_root, 'raw_window_data.pickle')

window_size = 128
window_overlap = 0.5

def load_label_data():
    label_data = []
    with open(label_dir, 'r') as label_file:
        for line in label_file:
            data = [int(x) for x in line.strip().split(' ')]
            label_info_dict = {
                'exp_id': data[0],
                'user_id': data[1],
                'label': data[2],
                'start_time': data[3],
                'end_time': data[4]
            }
            label_data.append(label_info_dict)
    return label_data


def get_raw_file_path(exp_id, user_id):
    acc_file_path = os.path.join(raw_dir, 'acc_exp{:02d}_user{:02d}.txt'.format(exp_id, user_id))
    gyro_file_path = os.path.join(raw_dir, 'gyro_exp{:02d}_user{:02d}.txt'.format(exp_id, user_id))
    return acc_file_path, gyro_file_path


def get_raw_data_chunk(label_data_dict):
    acc_file_path, gyro_file_path = get_raw_file_path(label_data_dict['exp_id'], label_data_dict['user_id'])
    with open(acc_file_path, 'r') as acc_file:
        acc_chunk_lines = acc_file.readlines()[label_data_dict['start_time']:label_data_dict['end_time'] + 1]
    with open(gyro_file_path, 'r') as gyro_file:
        gyro_chunk_lines = gyro_file.readlines()[label_data_dict['start_time']:label_data_dict['end_time'] + 1]
    acc_chunk_data = [[float(data) for data in line.split(' ')] for line in acc_chunk_lines]
    gyro_chunk_data = [[float(data) for data in line.split(' ')] for line in gyro_chunk_lines]

    return acc_chunk_data, gyro_chunk_data


def get_window_from_chunk(chunk_data):
    windows = []
    window_start_jump = int(window_size * (1 - window_overlap))
    for i in range(0, len(chunk_data), window_start_jump):
        if i + window_size >= len(chunk_data):
            break
        windows.append(chunk_data[i:i + window_size])
    return windows


def main():
    label_data = load_label_data()
    window_data = []
    for label_data_dict in tqdm(label_data):
        acc_chunk_data, gyro_chunk_data = get_raw_data_chunk(label_data_dict)
        acc_windows = get_window_from_chunk(acc_chunk_data)
        gyro_windows = get_window_from_chunk(gyro_chunk_data)
        assert len(acc_windows) == len(gyro_windows)
        for i in range(len(acc_windows)):
            window_data.append({
                'acc_window': acc_windows[i],
                'gyro_window': gyro_windows[i],
                'label': label_data_dict['label']
            })
    with open(output_pickle_path, 'wb') as output_pickle_file:
        output_pickle_file.write(pickle.dumps(window_data))


def test_raw_dataset():
    dataset = HaptRawDataset(output_pickle_path)
    print(dataset.__getitem__(0)[0].shape)

if __name__ == '__main__':
    # main()
    test_raw_dataset()