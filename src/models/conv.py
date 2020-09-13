import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param
from modules import Scaler


class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size, rate=1, track=False):
        super().__init__()
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
                  Scaler(rate),
                  nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                           Scaler(rate),
                           nn.BatchNorm2d(hidden_size[i + 1], momentum=None, track_running_stats=track),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten(),
                       nn.Linear(hidden_size[-1], classes_size)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        out = self.blocks(x)
        if 'label_split' in input:
            mask = torch.zeros(cfg['classes_size'], device=out.device)
            mask[input['label_split']] = 1
            out = out * mask
        output['score'] = out
        output['loss'] = F.cross_entropy(output['score'], input['label'], reduction='mean')
        return output


def conv(model_rate=1):
    data_shape = cfg['data_shape']
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['conv']['hidden_size']]
    classes_size = cfg['classes_size']
    scaler_rate = model_rate / cfg['global_model_rate']
    track = cfg['track']
    cfg['model'] = {}
    model = Conv(data_shape, hidden_size, classes_size, scaler_rate, track)
    model.apply(init_param)
    return model