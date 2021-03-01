import os

import numpy as np
import torch
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from datasets.hapt_dataset import HaptDataset, array_to_one_hot
from train import train_x_data_path, train_y_data_path, train_val_ratio, batch_size, num_workers
from models.hapt_mlp_model import HaptMlpModel
from utils.model_utils import load_checkpoint

checkpoint_dir = './checkpoints'
checkpoint_filename = 'checkpoint_mlp.pth.tar'


def plot_importance(attr_importance):
    attr_id = [i for i in range(len(attr_importance))]
    fig = plt.figure()
    plt.bar(attr_id, attr_importance)
    plt.show()


def main():
    config = {'input_size': 561, 'output_size': 12, 'print_freq': 100}
    model = HaptMlpModel(config).cuda()
    load_checkpoint(model, checkpoint_dir, checkpoint_filename)
    ig = IntegratedGradients(model)

    dataset = HaptDataset(train_x_data_path, train_y_data_path)
    train_length = int(len(dataset) * train_val_ratio)
    val_length = len(dataset) - train_length
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(7))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    attr_list = None
    for i, data in enumerate(val_loader):
        input, target = data
        input = input.cuda(non_blocking=True)
        target = array_to_one_hot(target, max_class=config['output_size'])
        target = torch.from_numpy(target)
        target = target.cuda(non_blocking=True)
        dummy_targe = torch.tensor(1)
        input.requires_grad_()
        attr, delta = ig.attribute(input, target=dummy_targe, return_convergence_delta=True)
        attr = attr.detach().cpu().numpy()
        if attr_list is not None:
            attr_list = np.vstack([attr_list, attr])
        else:
            attr_list = attr
    attr_list = np.asarray(attr_list)
    attr_avg_importance = attr_list.mean(axis=0)
    plot_importance(attr_avg_importance)




if __name__ == '__main__':
    main()