import copy
import numpy as np
import torch
from collections import OrderedDict


class Federation:
    def __init__(self, global_parameters, rate):
        self.global_parameters = global_parameters
        self.rate = rate
        self.num_models = int(1 / self.rate) ** 2
        self.idx = self.split_model()

    def split_model(self):
        idx_i = None
        idx = [OrderedDict() for _ in range(self.num_models)]
        output_weight = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            if parameter_type in ['weight', 'bias']:
                if parameter_type == 'weight':
                    if v.dim() > 1:
                        input_size = v.size(1)
                        output_size = v.size(0)
                        if idx_i is None:
                            idx_i = [torch.arange(input_size, device=v.device) for _ in range(self.num_models)]
                        input_idx_i = idx_i
                        if k == output_weight:
                            output_idx_i = [torch.arange(output_size, device=v.device) for _ in range(self.num_models)]
                            idx_i = output_idx_i
                        else:
                            half_output_idx_i = torch.chunk(torch.arange(output_size, device=v.device),
                                                            int(np.sqrt(self.num_models)))
                            output_idx_i = half_output_idx_i + half_output_idx_i[::-1]
                            idx_i = half_output_idx_i + half_output_idx_i
                        for m in range(self.num_models):
                            idx[m][k] = torch.meshgrid(output_idx_i[m], input_idx_i[m])
                    else:
                        input_idx_i = idx_i
                        for m in range(self.num_models):
                            idx[m][k] = input_idx_i[m]
                else:
                    input_idx_i = idx_i
                    for m in range(self.num_models):
                        idx[m][k] = input_idx_i[m]
        return idx

    def distribute(self, num_active_users):
        local_parameters = [OrderedDict() for _ in range(num_active_users)]
        model_idx = np.random.choice(range(self.num_models), num_active_users, replace=True)
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(model_idx)):
                if parameter_type in ['weight', 'bias']:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[self.idx[model_idx[m]][k]])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[self.idx[model_idx[m]][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[self.idx[model_idx[m]][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters, model_idx

    def combine(self, local_parameters, model_idx):
        count = OrderedDict()
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(local_parameters)):
                if parameter_type in ['weight', 'bias']:
                    tmp_v[self.idx[model_idx[m]][k]] += local_parameters[m][k]
                    count[k][self.idx[model_idx[m]][k]] += 1
                else:
                    tmp_v += local_parameters[m][k]
                    count[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        return