import torch
import torch.nn as nn

class HaptCnnModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(HaptCnnModel, self).__init__()
        self.cfg = cfg

        self.cnn_layer1 = nn.Conv1d(6, 18, kernel_size=7, stride=1)
        self.cnn_layer2 = nn.Conv1d(18, 54, kernel_size=7, stride=1)
        self.cnn_layer3 = nn.Conv1d(54, 162, kernel_size=7, stride=1)
        self.avg_pool_layer = nn.AvgPool1d(kernel_size=2, stride=2)
        self.output_layer = nn.Linear(8910, cfg['output_size'])

    def forward(self, x):
        x = self.cnn_layer1(x)
        x = torch.relu(x)
        x = self.cnn_layer2(x)
        x = torch.relu(x)
        x = self.cnn_layer3(x)
        x = torch.relu(x)
        x = self.avg_pool_layer(x).reshape(x.size()[0], -1)
        # print(x.shape)
        x = self.output_layer(x)
        return x