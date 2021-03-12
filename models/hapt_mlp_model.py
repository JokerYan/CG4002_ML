import torch
import torch.nn as nn

class HaptMlpModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(HaptMlpModel, self).__init__()
        self.cfg = cfg

        self.hidden_size = 256
        self.hidden_layer_count = 3
        self.input_layer = nn.Linear(cfg['input_size'], self.hidden_size)
        fc_list = [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.hidden_layer_count - 1)]
        self.fc_list = nn.ModuleList(fc_list)
        self.output_layer = nn.Linear(self.hidden_size, cfg['output_size'])
        self.intermediate_data = {}

    def forward(self, x):
        self.intermediate_data['input'] = x.detach().cpu().numpy()
        x = self.input_layer(x)
        x = torch.tanh(x)
        self.intermediate_data['hidden_1'] = x.detach().cpu().numpy()
        for i, fc_layer in enumerate(self.fc_list):
            x = fc_layer(x)
            x = torch.tanh(x)
            self.intermediate_data['hidden_{}'.format(i)] = x.detach().cpu().numpy()
        x = self.output_layer(x)
        x = torch.softmax(x, dim=1)
        self.intermediate_data['output'] = x.detach().cpu().numpy()
        return x