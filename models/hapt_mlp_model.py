import torch
import torch.nn as nn
import numpy as np
import pickle

class HaptMlpModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(HaptMlpModel, self).__init__()
        self.cfg = cfg
        self.relu = torch.nn.ReLU()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.hidden_size = 256
        self.hidden_layer_count = 3
        self.input_layer = nn.Linear(cfg['input_size'], self.hidden_size)
        fc_list = [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.hidden_layer_count - 1)]
        self.fc_list = nn.ModuleList(fc_list)
        self.output_layer = nn.Linear(self.hidden_size, cfg['output_size'])
        # self.intermediate_data = {}

        self.input_layer_scale = None
        self.fc_list_scale = [None for _ in range(self.hidden_layer_count - 1)]
        self.output_layer_scale = None
        self.input_layer_bias = None
        self.fc_list_bias = [None for _ in range(self.hidden_layer_count - 1)]
        self.output_layer_bias = None


    def forward(self, x):
        x = self.quant(x)
        # self.intermediate_data['input'] = x.detach().cpu().numpy()
        # last_x = x
        x = self.input_layer(x)
        # if x.dtype == torch.quint8 and self.input_layer_bias is not None:
        #     input_scale = self.input_layer_scale
        #     weight_scale = self.input_layer.weight().q_per_channel_scales()
        #     total_scale = np.array(input_scale * weight_scale, dtype=np.float64)
        #     output_scale = np.array(self.input_layer.scale, dtype=np.float32)
        #     output_zp = np.array(self.input_layer.zero_point, dtype=np.int32)
        #     # print(dir(self.input_layer))
        #     bias_array = np.array(self.input_layer_bias)
        #     int_x = np.array(last_x.int_repr(), dtype=np.int32) - 64
        #     weights = np.array(self.input_layer.weight().int_repr(), dtype=np.int32).transpose()
        #     output = np.matmul(int_x, weights) + bias_array
        #     # print(self.rescale(np.matmul(int_x, weights)))
        #     # print(bias_array)
        #     # print(self.input_layer.bias())
        #     print(np.round(output * total_scale / output_scale + output_zp))
        #     # print(last_x)
        #     # print(self.input_layer.weight())
        #     target_output = np.array(x.int_repr(), dtype=np.int32)
        #     print(target_output)
        #     # print(x)
        #     print("====================================")
        x = self.relu(x)
        # self.intermediate_data['hidden_1'] = x.detach().cpu().numpy()
        for i, fc_layer in enumerate(self.fc_list):
            x = fc_layer(x)
            x = self.relu(x)
            # self.intermediate_data['hidden_{}'.format(i)] = x.detach().cpu().numpy()
        x = self.output_layer(x)
        # self.intermediate_data['output'] = x.detach().cpu().numpy()
        x = self.dequant(x)
        return x

    def save_quantized_weights(self):
        # print(self.input_layer.weight())
        self.input_layer_bias = self.quantize_bias(self.input_layer.bias(), self.input_layer, self.quant.scale)
        self.fc_list_bias[0] = self.quantize_bias(self.fc_list[0].bias(), self.fc_list[0], self.input_layer.scale)
        self.fc_list_bias[1] = self.quantize_bias(self.fc_list[1].bias(), self.fc_list[1], self.fc_list[0].scale)
        self.output_layer_bias = self.quantize_bias(self.output_layer.bias(), self.output_layer, self.fc_list[1].scale)
        data = {
            'input_data_scale': self.quant.scale,
            'input_data_zero_point': self.quant.zero_point,
            'input_layer': {
                'weight': self.input_layer.weight().int_repr(),
                'bias': self.input_layer_bias,
                'scale_in': self.quant.scale,
                'scale_weight': self.input_layer.weight().q_per_channel_scales(),
                'scale_out': self.input_layer.scale,
                'zp_in': self.quant.zero_point,
                'zp_out': self.input_layer.zero_point,
            },
            'fc_layer_0': {
                'weight': self.fc_list[0].weight().int_repr(),
                'bias': self.fc_list_bias[0],
                'scale_in': self.input_layer.scale,
                'scale_weight': self.fc_list[0].weight().q_per_channel_scales(),
                'scale_out': self.fc_list[0].scale,
                'zp_in': self.input_layer.zero_point,
                'zp_out': self.fc_list[0].zero_point,
            },
            'fc_layer_1': {
                'weight': self.fc_list[1].weight().int_repr(),
                'bias': self.fc_list_bias[1],
                'scale_in': self.fc_list[0].scale,
                'scale_weight': self.fc_list[1].weight().q_per_channel_scales(),
                'scale_out': self.fc_list[1].scale,
                'zp_in': self.fc_list[0].zero_point,
                'zp_out': self.fc_list[1].zero_point,
            },
            'output_layer': {
                'weight': self.output_layer.weight().int_repr(),
                'bias': self.output_layer_bias,
                'scale_in': self.fc_list[1].scale,
                'scale_weight': self.output_layer.weight().q_per_channel_scales(),
                'scale_out': self.output_layer.scale,
                'zp_in': self.fc_list[1].zero_point,
                'zp_out': self.output_layer.zero_point,
            },
        }
        save_filename = 'quantized_weights.pickle'
        pickle.dump(data, open(save_filename, 'wb'))
        print('quantized weights saved to {}'.format(save_filename))

    def quantize_bias(self, fp_bias, layer, input_scale):
        weight_scales = layer.weight().q_per_channel_scales() * input_scale
        int_bias = torch.quantize_per_channel(fp_bias, weight_scales, weight_scales * 0, axis=0, dtype=torch.qint32)
        return int_bias.int_repr()