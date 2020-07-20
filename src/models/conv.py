import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size):
        super().__init__()
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
                  Normalization(hidden_size[0]),
                  Activation(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                           Normalization(hidden_size[i + 1]),
                           Activation(inplace=True),
                           nn.MaxPool2d(2)])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten(),
                       nn.Linear(hidden_size[-1], classes_size)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        output['label'] = self.blocks(x)
        output['loss'] = F.cross_entropy(output['label'], input['label'], reduction='mean')
        return output


def conv(mode='global'):
    data_shape = cfg['data_shape']
    hidden_size = cfg['conv'][mode]['hidden_size']
    classes_size = cfg['classes_size']
    cfg['model'] = {}
    model = Conv(data_shape, hidden_size, classes_size)
    model.apply(init_param)
    return model