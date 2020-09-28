from config import cfg
from data import fetch_dataset, make_data_loader
from utils import collate, process_dataset, ntuple
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import time

# class LocalConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
#                  active_rate=1.0):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         _pair = ntuple(2)
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.groups = groups
#         self.bias = bias
#         self.active_rate = active_rate
#         self.make_parameters(self.active_rate)
#
#     def make_parameters(self, active_rate):
#         self.active_rate = active_rate
#         self.active_channels = round(self.out_channels * self.active_rate)
#         if self.active_channels <= 0:
#             raise ValueError('Not valid active rate')
#         self.idle_channels = self.out_channels - self.active_channels
#         self.weight_active = nn.Parameter(
#             torch.Tensor(self.active_channels, self.in_channels // self.groups, *self.kernel_size))
#         self.register_buffer('weight_idle',
#                              torch.Tensor(self.idle_channels, self.in_channels // self.groups, *self.kernel_size))
#         if self.bias:
#             self.bias_active = nn.Parameter(torch.Tensor(self.active_channels))
#             self.register_buffer('bias_idle', torch.Tensor(self.idle_channels))
#         else:
#             self.register_parameter('bias_active', None)
#             self.register_parameter('bias_idle', None)
#         self.reset_parameters()
#         return
#
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight_active, a=math.sqrt(5))
#         if self.bias_active is not None:
#             fan_in_active, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_active)
#             bound_active = 1 / math.sqrt(fan_in_active)
#             nn.init.uniform_(self.bias_active, -bound_active, bound_active)
#         if self.idle_channels > 0:
#             nn.init.kaiming_uniform_(self.weight_idle, a=math.sqrt(5))
#             if self.bias_active is not None:
#                 fan_in_idle, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_idle)
#                 bound_idle = 1 / math.sqrt(fan_in_idle)
#                 nn.init.uniform_(self.bias_idle, -bound_idle, bound_idle)
#         return
#
#     def forward(self, input):
#         active_output = F.conv2d(input, self.weight_active, self.bias_active, self.stride,
#                                  self.padding, self.dilation, self.groups)
#         if self.idle_channels > 0:
#             print(self.weight_idle.size())
#             idle_output = F.conv2d(input, self.weight_idle, self.bias_idle, self.stride,
#                                    self.padding, self.dilation, self.groups)
#             output = torch.cat([active_output, idle_output], dim=1)
#         else:
#             output = active_output
#         return output
#
#
# if __name__ == '__main__':
#     # active_rate = 0.0625
#     active_rate = 1
#     num_layers = 50
#     input = torch.randn(100, 128, 32, 32)
#     model = []
#     for i in range(num_layers):
#         model.append(LocalConv2d(128, 128, 3, 1, 1, active_rate=active_rate))
#     model = nn.Sequential(*model)
#     output = model(input)
#     s = time.time()
#     output = model(input)
#     print(time.time()-s)
#     target = output.new_ones(output.size())
#     loss = F.mse_loss(output, target)
#     s = time.time()
#     loss.backward()
#     print(time.time() - s)
#     print(output.size())
#     # for name, param in model.named_parameters():
#     #     print(name, param.size(), param.grad.size())
#     # print(model.weight_idle.grad, model.bias_idle.grad)

# import itertools
#
# if __name__ == '__main__':
#     mode = ['a', 'b', 'c', 'd']
#     comb = []
#     for i in range(1, len(mode) + 1):
#         comb_i = [''.join(list(x)) for x in itertools.combinations(mode, i)]
#         comb.extend(comb_i)
#     print(comb)
#     print(len(comb))


if __name__ == '__main__':
    torch.manual_seed(4)
    N = 100
    C = 10
    c = 2
    x = torch.randn(N, 20)
    m = nn.Linear(20, C)
    optimizer = torch.optim.SGD(m.parameters(), lr=0.01)
    label = torch.arange(c).repeat(N // c)
    print(label)
    norm = m.weight.abs().mean(dim=1)
    print(norm)
    for i in range(100):
        y = m(x)
        mask = torch.zeros(C)
        mask[:c] = 1
        y = y.masked_fill(mask == 0, 0)
        # y = y.masked_fill(mask == 0, float('-inf'))
        # y_c, y_nc = y[:, :c], y[:, c:]
        # y = torch.cat([y_c, y_nc.detach()], dim=1)
        y = y.log_softmax(dim=-1)
        loss = F.nll_loss(y, label)
        loss.backward()
        optimizer.step()
    norm = m.weight.abs().mean(dim=1)
    print(loss)
    print(norm)