import torch


def cal_accuracy(output, target):
    with torch.no_grad():
        _, prediction = torch.max(output, dim=1)
        correct = (prediction == target).sum()
        return correct / len(prediction)