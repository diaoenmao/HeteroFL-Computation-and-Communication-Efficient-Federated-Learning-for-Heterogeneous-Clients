import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param
from modules import Scaler

Normalization = nn.BatchNorm1d
Activation = nn.ReLU


class MLP(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size, rate):
        super().__init__()
        blocks = [nn.Flatten(),
                  nn.Linear(np.prod(data_shape), hidden_size[0]),
                  Normalization(hidden_size[0], track_running_stats=False),
                  Activation(inplace=True),
                  Scaler(rate)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([nn.Linear(hidden_size[i], hidden_size[i + 1]),
                           Normalization(hidden_size[i + 1], track_running_stats=False),
                           Activation(inplace=True),
                           Scaler(rate)])
        blocks.extend([nn.Linear(hidden_size[-1], classes_size)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        output['label'] = self.blocks(x)
        output['loss'] = F.cross_entropy(output['label'], input['label'], reduction='mean')
        return output


def mlp(rate=1):
    data_shape = cfg['data_shape']
    hidden_size = [int(np.ceil(rate * x)) for x in cfg['mlp']['hidden_size']]
    classes_size = cfg['classes_size']
    cfg['model'] = {}
    model = MLP(data_shape, hidden_size, classes_size, rate)
    model.apply(init_param)
    return model