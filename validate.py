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
from utils.model_utils import save_checkpoint, load_checkpoint, quantize_model
from utils.transforms import Float16ToInt8
from utils.feature_set import feature_set_A, feature_set_B

train_x_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Train\X_train.txt"
train_y_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Train\y_train.txt"
test_x_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Test\X_test.txt"
test_y_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Test\y_test.txt"

val_length = 1000
batch_size = 16
num_workers = 4
total_epoch = 15
learning_rate = 0.0001
target_features = []

model_name = 'mlp'
device = 'cpu'

def main():
    if model_name == 'mlp' or model_name == 'knn':
        train_feature_data_list = get_collected_data(val=False)
        train_dataset = DanceDataset(train_feature_data_list)
        val_feature_data_list = get_collected_data(val=True)
        val_dataset = DanceDataset(val_feature_data_list)
        # dataset = HaptDataset(train_x_data_path, train_y_data_path, target_features=target_features)
    elif model_name == 'cnn':
        dataset = HaptRawDataset(raw_json_path)
    # train_length = len(dataset) - val_length
    # train_dataset, val_dataset = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(7))


    model = None
    if model_name == 'mlp':
        input_size = train_dataset.input_size()
        config = {'input_size': input_size, 'output_size': 12, 'print_freq': 100}
        model = HaptMlpModel(config)
        load_checkpoint(model, './checkpoints', 'best_mlp.pth.tar')
        model = quantize_model(model, train_dataset)
        model = model.to(device)
    if model_name == 'cnn':
        config = {'output_size': 12, 'print_freq': 100}
        model = HaptCnnModel(config)
        load_checkpoint(model, './checkpoints', 'best_cnn.pth.tar')
        model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    summary = SummaryWriter()
    val_acc, val_f1 = validate(config, val_loader, model, criterion, 0, summary, print_output=False, device=device)
    print('Val acc:{}'.format(val_acc))
    print('Val f1:{}'.format(val_f1))


if __name__ == '__main__':
    main()