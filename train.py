import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier

from datasets.hapt_dataset import HaptDataset
from datasets.hapt_raw_dataset import HaptRawDataset
from core.function import train, validate
from models.hapt_mlp_model import HaptMlpModel
from models.hapt_cnn_model import HaptCnnModel
from utils.hapt_raw_data_processing import output_json_path as raw_json_path
from utils.model_utils import save_checkpoint
from utils.transforms import Float16ToInt8

train_x_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Train\X_train.txt"
train_y_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Train\y_train.txt"
test_x_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Test\X_test.txt"
test_y_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Test\y_test.txt"

train_val_ratio = 0.8
batch_size = 16
num_workers = 4
total_epoch = 100
learning_rate = 0.0001

model_name = 'cnn'

def main():
    if model_name == 'mlp' or model_name == 'knn':
        dataset = HaptDataset(train_x_data_path, train_y_data_path)
        test_dataset = HaptDataset(test_x_data_path, test_y_data_path)
        train_length = int(len(dataset) * train_val_ratio)
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(7))

        # mean, std = dataset.get_mean_std(train_dataset.indices)
        # dataset.normalize(mean, std)
        # test_dataset.normalize(mean, std)

        if model_name == 'mlp':
            train_mlp(train_dataset, val_dataset, test_dataset)
        elif model_name == 'knn':
            train_knn(train_dataset, val_dataset, test_dataset)

    if model_name == 'cnn':
        dataset = HaptRawDataset(raw_json_path, Float16ToInt8())
        train_length = int(len(dataset) * train_val_ratio)
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(7))
        train_cnn(train_dataset, val_dataset)


def train_knn(train_dataset, val_dataset, test_dataset):
    knn = KNeighborsClassifier(n_neighbors=20, weights="uniform")
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
    test_x = []
    test_y = []
    for i, data in enumerate(test_dataset):
        input, target = data
        test_x.append(np.asarray(input))
        test_y.append(np.asarray(target))
    test_y = np.asarray(test_y).reshape(-1)

    knn.fit(train_x, train_y)
    val_output = knn.predict(val_x)
    val_output, val_y = np.asarray(val_output), np.asarray(val_y)
    correct = (val_output == val_y).sum()
    val_acc = correct / len(val_output)
    print("Val Accuracy: ", val_acc)

    test_output = knn.predict(test_x)
    test_output, test_y = np.asarray(test_output), np.asarray(test_y)
    correct = (test_output == test_y).sum()
    test_acc = correct / len(test_output)
    print("Test Accuracy: ", test_acc)


def train_cnn(train_dataset, val_dataset):
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    config = {'output_size': 12, 'print_freq': 100}
    model = HaptCnnModel(config).cuda()
    model = nn.DataParallel(model).cuda()
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
    for epoch in range(total_epoch):
        train(config, train_loader, model, criterion, optimizer, epoch, summary)
        validate(config, val_loader, model, criterion, epoch, summary)


def train_mlp(train_dataset, val_dataset, test_dataset):
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    config = {'input_size': 561, 'output_size': 12, 'print_freq': 100}
    model = HaptMlpModel(config).cuda()
    model = nn.DataParallel(model).cuda()
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
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    summary = SummaryWriter()
    for epoch in range(total_epoch):
        train(config, train_loader, model, criterion, optimizer, epoch, summary)
        validate(config, val_loader, model, criterion, epoch, summary)
        validate(config, test_loader, model, criterion, epoch, summary, "test")
        save_checkpoint(model, epoch, optimizer, './checkpoints', 'checkpoint_mlp.pth.tar')


if __name__ == "__main__":
    main()