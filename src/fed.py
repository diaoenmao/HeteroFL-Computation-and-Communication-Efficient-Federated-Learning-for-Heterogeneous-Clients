import copy
import torch
import numpy as np
from config import cfg
from collections import OrderedDict


class Federation:
    def __init__(self, global_parameters, rate, label_split):
        self.global_parameters = global_parameters
        self.rate = rate
        self.label_split = label_split
        self.make_model_rate()

    def make_model_rate(self):
        if cfg['model_split_mode'] == 'dynamic':
            rate_idx = torch.multinomial(torch.tensor(cfg['proportion']), num_samples=cfg['num_users'],
                                         replacement=True).tolist()
            self.model_rate = np.array(self.rate)[rate_idx]
        elif cfg['model_split_mode'] == 'fix':
            self.model_rate = np.array(self.rate)
        else:
            raise ValueError('Not valid model split mode')
        return

    def split_model(self, user_idx):
        if cfg['model_name'] == 'conv':
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
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
                            if k == output_bias_name:
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
                    if 'weight' in parameter_type or 'bias' in parameter_type:
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
        elif cfg['model_name'] == 'transformer':
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if 'embedding' in k.split('.')[-2]:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_input_size = int(np.ceil(input_size * scaler_rate))
                                    input_idx_i_m = torch.arange(input_size, device=v.device)[:local_input_size]
                                    idx_i[m] = input_idx_i_m
                                elif 'decoder' in k and 'linear2' in k:
                                    input_idx_i_m = idx_i[m]
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                    * scaler_rate))
                                    output_idx_i_m = (torch.arange(output_size, device=v.device).reshape(
                                        cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                    idx_i[m] = output_idx_i_m
                                else:
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                    idx_i[m] = output_idx_i_m
                                idx[m][k] = (output_idx_i_m, input_idx_i_m)
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            input_size = v.size(0)
                            if 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = torch.arange(input_size, device=v.device)
                                idx[m][k] = input_idx_i_m
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                                if 'linear_v' not in k:
                                    idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
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
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters, param_idx

    def combine(self, local_parameters, param_idx, user_idx):
        count = OrderedDict()
        if cfg['model_name'] == 'conv':
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if k == output_weight_name:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if k == output_bias_name:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif 'resnet' in cfg['model_name']:
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if 'linear' in k:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if 'linear' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif cfg['model_name'] == 'transformer':
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                if k.split('.')[-2] == 'embedding':
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                elif 'decoder' in k and 'linear2' in k:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if 'decoder' in k and 'linear2' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        else:
            raise ValueError('Not valid model name')

        return