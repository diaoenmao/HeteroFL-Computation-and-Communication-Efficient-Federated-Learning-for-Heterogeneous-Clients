import copy
import torch
import numpy as np
from config import cfg
from collections import OrderedDict


class Federation:
    def __init__(self, global_parameters, rate):
        self.global_parameters = global_parameters
        self.rate = rate
        self.make_model_rate()

    def make_model_rate(self):
        if cfg['model_split_mode'] == 'dynamic':
            rate_idx = np.random.choice(np.arange(len(self.rate)), cfg['num_users'], p=cfg['proportion'])
            self.model_rate = np.array(self.rate)[rate_idx]
        elif cfg['model_split_mode'] == 'fix':
            self.model_rate = np.array(self.rate)
        else:
            raise ValueError('Not valid model split mode')
        return

    def split_model(self, user_idx):
        if cfg['model_name'] in ['conv', 'mlp']:
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if parameter_type in ['weight', 'bias', 'running_mean', 'running_var']:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                if k == output_weight_name:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                else:
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                idx[m][k] = output_idx_i_m, input_idx_i_m
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        pass
        elif 'resnet' in cfg['model_name']:
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if parameter_type in ['weight', 'bias', 'running_mean', 'running_var']:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if 'conv1' in k or 'conv2' in k:
                                    if idx_i[m] is None:
                                        idx_i[m] = torch.arange(input_size, device=v.device)
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                    idx_i[m] = output_idx_i_m
                                elif 'shortcut' in k:
                                    input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                    output_idx_i_m = idx_i[m]
                                elif 'linear' in k:
                                    input_idx_i_m = idx_i[m]
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                else:
                                    raise ValueError('Not valid k')
                                idx[m][k] = (output_idx_i_m, input_idx_i_m)
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            input_size = v.size(0)
                            if 'linear' in k:
                                input_idx_i_m = torch.arange(input_size, device=v.device)
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        else:
            raise ValueError('Not valid model name')
        return idx

    def distribute(self, user_idx):
        self.make_model_rate()
        param_idx = self.split_model(user_idx)
        local_parameters = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if parameter_type in ['weight', 'bias', 'running_mean', 'running_var']:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters, param_idx

    def combine(self, local_parameters, param_idx):
        count = OrderedDict()
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(local_parameters)):
                if parameter_type in ['weight', 'bias', 'running_mean', 'running_var']:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                            count[k][torch.meshgrid(param_idx[m][k])] += 1
                        else:
                            tmp_v[param_idx[m][k]] += local_parameters[m][k]
                            count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v[param_idx[m][k]] += local_parameters[m][k]
                        count[k][param_idx[m][k]] += 1
                else:
                    tmp_v += local_parameters[m][k]
                    count[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        return