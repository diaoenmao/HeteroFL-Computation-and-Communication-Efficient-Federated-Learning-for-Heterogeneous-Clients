import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        if self.training:
            output = input / self.rate
        else:
            output = input
        return output