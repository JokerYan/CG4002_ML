import os
import torch


def save_checkpoint(model, epoch, optimizer, output_dir, filename='checkpoint.pth.tar'):
    states = {
        'epoch': epoch,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(states, os.path.join(output_dir, filename))
    print('model saved to {}'.format(os.path.join(output_dir, filename)))


def load_checkpoint(model, output_dir, filename='checkpoint.pth.tar'):
    states = torch.load(os.path.join(output_dir, filename))
    model.load_state_dict(states['state_dict'])
    print('model loaded from {}'.format(os.path.join(output_dir, filename)))

