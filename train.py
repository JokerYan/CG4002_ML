import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from datasets.dance_dataset import DanceDataset
from datasets.hapt_dataset import HaptDataset
from datasets.hapt_raw_dataset import HaptRawDataset
from core.function import train, validate
from models.hapt_mlp_model import HaptMlpModel
from models.hapt_cnn_model import HaptCnnModel
from utils.collected_data_process import get_collected_data
from utils.hapt_raw_data_processing import output_pickle_path as raw_json_path
from utils.model_utils import save_checkpoint
from utils.transforms import Float16ToInt8
from utils.feature_set import feature_set_A, feature_set_B

train_x_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Train\X_train.txt"
train_y_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Train\y_train.txt"
test_x_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Test\X_test.txt"
test_y_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Test\y_test.txt"

use_k_fold = False
split_k = 4
batch_size = 16
num_workers = 4
total_epoch = 100
learning_rate = 0.0001
target_features = []

model_name = 'mlp'


def main():
    # fold_dataset_list = split_k_fold(split_k)
    val_acc_sum = 0
    val_f1_sum = 0
    for val_idx in range(split_k):
        # fold_train_dataset = ConcatDataset(fold_dataset_list[:val_idx] + fold_dataset_list[val_idx + 1:])
        # fold_val_dataset = fold_dataset_list[val_idx]
        # val_acc, val_f1 = train_single_fold(fold_train_dataset, fold_val_dataset)

        train_feature_data_list = get_collected_data(val=False)
        train_dataset = DanceDataset(train_feature_data_list)
        val_feature_data_list = get_collected_data(val=True)
        val_dataset = DanceDataset(val_feature_data_list)

        val_acc, val_f1 = train_single_fold(train_dataset, val_dataset)
        val_acc_sum += val_acc
        val_f1_sum += val_f1
        if not use_k_fold:
            break
    val_acc_avg = val_acc_sum / split_k if use_k_fold else val_acc_sum
    val_f1_avg = val_f1_sum / split_k if use_k_fold else val_f1_sum
    print("========================================")
    print("Results from k={} fold cross validation:".format(split_k))
    print("Validation Accuracy: {}".format(val_acc_avg))
    print("Validation F1 Score: {}".format(val_f1_avg))


def split_k_fold(k):
    dataset = None
    if model_name == 'mlp' or model_name == 'knn':
        feature_data_list = get_collected_data()
        dataset = DanceDataset(feature_data_list)
        # dataset = HaptDataset(train_x_data_path, train_y_data_path, target_features=target_features)
        # test_dataset = HaptDataset(test_x_data_path, test_y_data_path, target_features=target_features)
    elif model_name == 'cnn':
        dataset = HaptRawDataset(raw_json_path)
        # dataset = HaptRawDataset(raw_json_path, Float16ToInt8())
    fold_length = len(dataset) // k
    fold_length_list = [fold_length for _ in range(k-1)] + [len(dataset) - fold_length * (k - 1)]
    fold_dataset_list = random_split(dataset, fold_length_list, generator=torch.Generator().manual_seed(7))
    return fold_dataset_list


def train_single_fold(train_dataset, val_dataset):
    if model_name == 'mlp':
        return train_mlp(train_dataset, val_dataset)
    elif model_name == 'knn':
        return train_knn(train_dataset, val_dataset)
    elif model_name == 'cnn':
        return train_cnn(train_dataset, val_dataset)


def train_knn(train_dataset, val_dataset):
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    train_x = []
    train_y = []
    for i, data in enumerate(train_dataset):
        input, target = data
        train_x.append(np.asarray(input))
        train_y.append(np.asarray(target))
    train_y = np.asarray(train_y).reshape(-1)
    val_x = []
    val_y = []
    for i, data in enumerate(val_dataset):
        input, target = data
        val_x.append(np.asarray(input))
        val_y.append(np.asarray(target))
    val_y = np.asarray(val_y).reshape(-1)

    knn.fit(train_x, train_y)
    val_output = knn.predict(val_x)
    val_output, val_y = np.asarray(val_output), np.asarray(val_y)
    val_acc = accuracy_score(val_output, val_y)
    val_f1 = f1_score(val_output, val_y, average='macro')
    print(confusion_matrix(val_output, val_y))
    # correct = (val_output == val_y).sum()
    # val_acc = correct / len(val_output)
    # print("Val Accuracy: ", val_acc)

    return val_acc, val_f1


def train_cnn(train_dataset, val_dataset):
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    config = {'output_size': 12, 'print_freq': 100}
    model = HaptCnnModel(config).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(
        # model.parameters(),
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    summary = SummaryWriter()
    best_val_acc = 0
    best_val_f1 = 0
    for epoch in range(total_epoch):
        train(config, train_loader, model, criterion, optimizer, epoch, summary)
        val_acc, val_f1 = validate(config, val_loader, model, criterion, epoch, summary)
        save_checkpoint(model, epoch, optimizer, './checkpoints', 'checkpoint_cnn.pth.tar')
        if val_f1 > best_val_f1:
            save_checkpoint(model, epoch, optimizer, './checkpoints', 'best_cnn.pth.tar')
            best_val_acc = val_acc
            best_val_f1 = val_f1
    return best_val_acc, best_val_f1


def train_mlp(train_dataset, val_dataset):
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    input_size = train_dataset.input_size()
    config = {'input_size': input_size, 'output_size': 12, 'print_freq': 100}
    model = HaptMlpModel(config).cuda()

    weight_tensor = [1 for _ in range(config['output_size'] - 2)] + [10, 10]
    criterion = nn.CrossEntropyLoss(torch.Tensor(weight_tensor)).cuda()
    # criterion = nn.CrossEntropyLoss(torch.Tensor([0.1, 0.1, 0.1, 0.1])).cuda()
    optimizer = torch.optim.Adam(
        # model.parameters(),
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    summary = SummaryWriter()
    best_val_acc = 0
    best_val_f1 = 0
    for epoch in range(total_epoch):
        train(config, train_loader, model, criterion, optimizer, epoch, summary)
        val_acc, val_f1 = validate(config, val_loader, model, criterion, epoch, summary)
        save_checkpoint(model, epoch, optimizer, './checkpoints', 'checkpoint_mlp.pth.tar')
        if val_f1 > best_val_f1:
            save_checkpoint(model, epoch, optimizer, './checkpoints', 'best_mlp.pth.tar')
            best_val_acc = val_acc
            best_val_f1 = val_f1
    return best_val_acc, best_val_f1


if __name__ == "__main__":
    main()