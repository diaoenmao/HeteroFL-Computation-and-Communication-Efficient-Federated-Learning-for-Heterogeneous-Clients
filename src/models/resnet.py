'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param
from config import cfg
from modules import Scaler


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, rate):
        super(Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.scaler = Scaler(rate)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.scaler(F.relu(self.bn1(x)))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.scaler(F.relu(self.bn2(out))))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.scaler = Scaler(rate)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.scaler(F.relu(self.bn1(x)))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.scaler(F.relu(self.bn2(out))))
        out = self.conv3(self.scaler(F.relu(self.bn3(out))))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, num_classes, rate):
        super(ResNet, self).__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1, rate=rate)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2, rate=rate)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2, rate=rate)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2, rate=rate)
        self.linear = nn.Linear(hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        output = {}
        x = input['img']
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        output['score'] = out
        output['loss'] = F.cross_entropy(output['score'], input['label'])
        return output


def resnet18(rate=1):
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(rate * x)) for x in cfg['resnet']['hidden_size']]
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], classes_size, rate)
    model.apply(init_param)
    return model


def resnet34(rate=1):
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(rate * x)) for x in cfg['resnet']['hidden_size']]
    model = ResNet(data_shape, hidden_size, Block, [3, 4, 6, 3], classes_size, rate)
    model.apply(init_param)
    return model


def resnet50(rate=1):
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(rate * x)) for x in cfg['resnet']['hidden_size']]
    model = ResNet(data_shape, hidden_size, Bottleneck, [3, 4, 6, 3], classes_size, rate)
    model.apply(init_param)
    return model


def resnet101(rate=1):
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(rate * x)) for x in cfg['resnet']['hidden_size']]
    model = ResNet(data_shape, hidden_size, Bottleneck, [3, 4, 23, 3], classes_size, rate)
    model.apply(init_param)
    return model


def resnet152(rate=1):
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = [int(np.ceil(rate * x)) for x in cfg['resnet']['hidden_size']]
    model = ResNet(data_shape, hidden_size, Bottleneck, [3, 8, 36, 3], classes_size, rate)
    model.apply(init_param)
    return model