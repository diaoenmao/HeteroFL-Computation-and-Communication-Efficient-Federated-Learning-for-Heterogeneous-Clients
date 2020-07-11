import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param

Normalization = nn.BatchNorm1d
Activation = nn.ReLU


class MLP(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size):
        super().__init__()
        blocks = [nn.Flatten(),
                  nn.Linear(np.prod(data_shape), hidden_size[0]),
                  Normalization(hidden_size[0]),
                  Activation(inplace=True)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([nn.Linear(hidden_size[i], hidden_size[i + 1]),
                           Normalization(hidden_size[i + 1]),
                           Activation(inplace=True)])
        blocks.extend([nn.Linear(hidden_size[-1], classes_size)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        output['label'] = self.blocks(x)
        output['loss'] = F.cross_entropy(output['label'], input['label'], reduction='mean')
        return output


def mlp():
    data_shape = cfg['data_shape']
    hidden_size = cfg['mlp']['hidden_size']
    classes_size = cfg['classes_size']
    cfg['model'] = {}
    model = MLP(data_shape, hidden_size, classes_size)
    model.apply(init_param)
    return model