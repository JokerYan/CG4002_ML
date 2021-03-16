import os
import torch


def save_checkpoint(model, epoch, optimizer, output_dir, filename='checkpoint.pth.tar'):
    states = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(states, os.path.join(output_dir, filename))
    print('model saved to {}'.format(os.path.join(output_dir, filename)))


# def save_checkpoint(model, output_dir, filename):
#     states = {
#         'state_dict': model.state_dict(),
#     }
#     torch.save(states, os.path.join(output_dir, filename))
#     print('model saved to {}'.format(os.path.join(output_dir, filename)))


def load_checkpoint(model, output_dir, filename='checkpoint.pth.tar'):
    states = torch.load(os.path.join(output_dir, filename))
    model.load_state_dict(states['state_dict'])
    print('model loaded from {}'.format(os.path.join(output_dir, filename)))


def quantize_model(model, dataset):
    print('model quantization started')
    backend = "fbgemm"
    model.eval()
    model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    # model = torch.quantization.fuse_modules(model, [['linear', 'relu']])
    model = torch.quantization.prepare(model, inplace=False)
    for i in range(100):
        cali_input = torch.from_numpy(dataset[i][0])  # calibration
        model(cali_input)
    model = torch.quantization.convert(model, inplace=False)
    print('model quantization finished')
    # print(torch.from_numpy(dataset[0][0]))
    # model(torch.from_numpy(dataset[0][0]).reshape([1, -1]))  # used to get input scale at each layer
    # print(dir(model.quant))
    # print(model.input_layer.weight())
    model.save_quantized_weights()
    return model