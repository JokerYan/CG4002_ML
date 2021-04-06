import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

data_root = r'C:\Code\Projects\Dance\CG4002\Ultra96\data'
# target_files = ['4_left_1616755739.3632548', '4_right_1616755739.405899', '1_hair_1615628068.944603']
target_files = ['1_left_1617440502.3576574', '1_right_1617440502.4122562']

max_time = 240
selected_sensor = [0, 1, 2]

def cvt_to_fft(data):
    N = 10
    T = 1 / 20
    yf = fft(data)
    xf = fftfreq(N, T)[:N // 2]
    yf = 2.0 / N * np.abs(yf[0:N // 2])
    return xf, yf

target_data = []
for filename in target_files:
    file_path = os.path.join(data_root, filename)
    target_data.append(pickle.load(open(file_path, 'rb')))

# velocity = np.zeros_like(target_data)
# for idx, data in enumerate(target_data):
#     vel_data = np.zeros_like(data, dtype=np.int32)
#     for i in range(len(data)):
#         for j in range(len(data[i])):
#             vel_data[i][j] = np.sum(data[i][:j+1])  # 20Hz
#     velocity[idx] = vel_data

# for idx in range(len(target_data)):
#     target_data[idx] = target_data[idx] - np.mean(target_data[idx], axis=1)[:, None]

fig, axs = plt.subplots(len(target_data))
for i, data in enumerate(target_data):
    sample_time = min(data.shape[1], max_time)
    x = [i for i in range(sample_time)]
    for j in selected_sensor:
        axs[i].plot(x, target_data[i][j][:sample_time])
        # axs[i].plot(x, velocity[i][j][:sample_time])
        axs[i].grid()
plt.show()

window_size = 50
window_count = 4
window_offset = 5  # discard first few windows
fig, axs = plt.subplots(len(target_data), window_count)
for i, data in enumerate(target_data):
    for j in selected_sensor:
        for k in range(window_count):
            xf, yf = cvt_to_fft(target_data[i][j][(k+window_offset)*window_size:(k+1+window_offset)*window_size])
            axs[i][k].plot(xf, yf)
            # axs[i].plot(x, velocity[i][j][:sample_time])
            axs[i][k].grid()
plt.show()