import torch
import time
import numpy as np
import pickle

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from core.evaluate import cal_accuracy

def train(cfg, train_loader, model, criterion, optimizer, epoch,
          summary, device='cuda'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        input, target = data
        data_time.update(time.time() - end)

        input = input.to(device)
        output = model(input)
        target = target.to(device).reshape(-1)

        loss = criterion(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # evaluation
        acc = cal_accuracy(output, target)
        accuracy.update(acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % cfg['print_freq'] == 0 or i == len(train_loader) - 1:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, accuracy=accuracy)
            print(msg)

    # write summary
    summary.add_scalar("Loss/train", losses.avg, epoch)
    summary.add_scalar("Acc/train", accuracy.avg, epoch)


def validate(cfg, val_loader, model, criterion, epoch, summary,
             phase="val", print_output=False, device='cuda'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.eval()
    end = time.time()
    targets = np.asarray([])
    outputs = np.asarray([])
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            input, target = data
            targets = np.append(targets, target)
            if print_output:
                print(input)

            input = input.to(device)
            target = target.to(device).reshape(-1)
            data_time.update(time.time() - end)

            output = model(input)
            # intermediate_data = model.intermediate_data
            # pickle.dump(intermediate_data, open('intermediate_data.pickle', 'wb+'))
            outputs = np.append(outputs, torch.argmax(output.detach().cpu(), dim=1).numpy())
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # evaluation
            acc = cal_accuracy(output, target)
            accuracy.update(acc)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % cfg['print_freq'] == 0 or i == len(val_loader) - 1:
                msg = '--- {phase:s} Data:\t' \
                      'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'.format(
                          epoch, i, len(val_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val if batch_time.val != 0 else float('inf'),
                          data_time=data_time, loss=losses, accuracy=accuracy, phase=phase)
                print(msg)

    # write summary
    summary.add_scalar("Loss/" + phase, losses.avg, epoch)
    summary.add_scalar("Acc/" + phase, accuracy.avg, epoch)

    outputs = outputs.reshape(-1)
    targets = targets.reshape(-1)
    val_acc = accuracy_score(outputs, targets)
    val_f1 = f1_score(outputs, targets, average='macro')
    if print_output:
        print(outputs)
    print(confusion_matrix(outputs, targets))
    return val_acc, val_f1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count