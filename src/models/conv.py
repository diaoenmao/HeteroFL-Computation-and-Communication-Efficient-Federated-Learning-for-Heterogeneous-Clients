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
        if cfg['norm'] == 'bn':
            norm = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)
        elif cfg['norm'] == 'in':
            norm = nn.GroupNorm(hidden_size[0], hidden_size[0])
        elif cfg['norm'] == 'ln':
            norm = nn.GroupNorm(1, hidden_size[0])
        elif cfg['norm'] == 'gn':
            norm = nn.GroupNorm(4, hidden_size[0])
        elif cfg['norm'] == 'none':
            norm = nn.Identity()
        else:
            raise ValueError('Not valid norm')
        if cfg['scale']:
            scaler = Scaler(rate)
        else:
            scaler = nn.Identity()
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
                  scaler,
                  norm,
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            if cfg['norm'] == 'bn':
                norm = nn.BatchNorm2d(hidden_size[i + 1], momentum=None, track_running_stats=track)
            elif cfg['norm'] == 'in':
                norm = nn.GroupNorm(hidden_size[i + 1], hidden_size[i + 1])
            elif cfg['norm'] == 'ln':
                norm = nn.GroupNorm(1, hidden_size[i + 1])
            elif cfg['norm'] == 'gn':
                norm = nn.GroupNorm(4, hidden_size[i + 1])
            elif cfg['norm'] == 'none':
                norm = nn.Identity()
            else:
                raise ValueError('Not valid norm')
            if cfg['scale']:
                scaler = Scaler(rate)
            else:
                scaler = nn.Identity()
            blocks.extend([nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                           scaler,
                           norm,
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
        if 'label_split' in input and cfg['mask']:
            label_mask = torch.zeros(cfg['classes_size'], device=out.device)
            label_mask[input['label_split']] = 1
            out = out.masked_fill(label_mask == 0, 0)
        output['score'] = out
        output['loss'] = F.cross_entropy(out, input['label'], reduction='mean')
        return output


def conv(model_rate=1, track=False):
    data_shape = cfg['data_shape']
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['conv']['hidden_size']]
    classes_size = cfg['classes_size']
    scaler_rate = model_rate / cfg['global_model_rate']
    model = Conv(data_shape, hidden_size, classes_size, scaler_rate, track)
    model.apply(init_param)
    return model