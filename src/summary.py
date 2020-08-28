import argparse
from collections import OrderedDict
import models
import os
from config import cfg
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import fetch_dataset, make_data_loader
from utils import makedir_exist_ok, to_device, process_control, process_dataset, collate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['batch_size'] = {'train': 2, 'test': 2}
rate = 1

def main():
    process_control()
    runExperiment()
    return


def runExperiment():
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset['train'])
    data_loader = make_data_loader(dataset)
    model = eval('models.{}(rate=rate).to(cfg["device"])'.format(cfg['model_name']))
    summary = summarize(data_loader['train'], model)
    content = parse_summary(summary)
    print(content)
    return


def make_size(input):
    if isinstance(input, tuple):
        return make_size(input[0])
    else:
        return list(input[0].size())


def summarize(data_loader, model):
    def register_hook(module):

        def hook(module, input, output):
            module_name = str(module.__class__.__name__)
            if module_name not in summary['count']:
                summary['count'][module_name] = 1
            else:
                summary['count'][module_name] += 1
            key = str(hash(module))
            if key not in summary['module']:
                summary['module'][key] = OrderedDict()
                summary['module'][key]['module_name'] = '{}_{}'.format(module_name, summary['count'][module_name])
                summary['module'][key]['input_size'] = []
                summary['module'][key]['output_size'] = []
                summary['module'][key]['params'] = {}
            input_size = make_size(input)
            output_size = make_size(output)
            summary['module'][key]['input_size'].append(input_size)
            summary['module'][key]['output_size'].append(output_size)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    if name in ['weight', 'weight_orig']:
                        if name not in summary['module'][key]['params']:
                            summary['module'][key]['params']['weight'] = {}
                            summary['module'][key]['params']['weight']['size'] = list(param.size())
                            summary['module'][key]['coordinates'] = []
                            summary['module'][key]['params']['weight']['mask'] = torch.zeros(
                                summary['module'][key]['params']['weight']['size'], dtype=torch.long,
                                device=cfg['device'])
                    elif name == 'bias':
                        if name not in summary['module'][key]['params']:
                            summary['module'][key]['params']['bias'] = {}
                            summary['module'][key]['params']['bias']['size'] = list(param.size())
                            summary['module'][key]['params']['bias']['mask'] = torch.zeros(
                                summary['module'][key]['params']['bias']['size'], dtype=torch.long,
                                device=cfg['device'])
                    else:
                        continue
            if len(summary['module'][key]['params']) == 0:
                return
            if 'weight' in summary['module'][key]['params']:
                weight_size = summary['module'][key]['params']['weight']['size']
                summary['module'][key]['coordinates'].append(
                    [torch.arange(weight_size[i], device=cfg['device']) for i in range(len(weight_size))])
            else:
                raise ValueError('Not valid parametrized module')
            for name in summary['module'][key]['params']:
                coordinates = summary['module'][key]['coordinates'][-1]
                if name == 'weight':
                    if len(coordinates) == 1:
                        summary['module'][key]['params'][name]['mask'][coordinates[0]] += 1
                    elif len(coordinates) >= 2:
                        summary['module'][key]['params'][name]['mask'][
                            coordinates[0].view(-1, 1), coordinates[1].view(1, -1),] += 1
                    else:
                        raise ValueError('Not valid coordinates dimension')
                elif name == 'bias':
                    if len(coordinates) == 1:
                        summary['module'][key]['params'][name]['mask'] += 1
                    elif len(coordinates) >= 2:
                        summary['module'][key]['params'][name]['mask'] += 1
                    else:
                        raise ValueError('Not valid coordinates dimension')
                else:
                    raise ValueError('Not valid parameters type')
            return

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) \
                and not isinstance(module, nn.ModuleDict) and module != model:
            hooks.append(module.register_forward_hook(hook))
        return

    run_mode = True
    summary = OrderedDict()
    summary['module'] = OrderedDict()
    summary['count'] = OrderedDict()
    hooks = []
    model.train(run_mode)
    model.apply(register_hook)
    for i, input in enumerate(data_loader):
        input = collate(input)
        input = to_device(input, cfg['device'])
        model(input)
        break
    for h in hooks:
        h.remove()
    summary['total_num_param'] = 0
    for key in summary['module']:
        num_params = 0
        for name in summary['module'][key]['params']:
            num_params += (summary['module'][key]['params'][name]['mask'] > 0).sum().item()
        summary['total_num_param'] += num_params
    summary['total_space_param'] = abs(summary['total_num_param'] * 32. / 8 / (1024 ** 2.))
    return summary


def parse_summary(summary):
    content = ''
    headers = ['Module Name', 'Input Size', 'Weight Size', 'Output Size', 'Number of Parameters']
    records = []
    for key in summary['module']:
        if 'weight' not in summary['module'][key]['params']:
            continue
        module_name = summary['module'][key]['module_name']
        input_size = str(summary['module'][key]['input_size'])
        weight_size = str(summary['module'][key]['params']['weight']['size']) if (
                'weight' in summary['module'][key]['params']) else 'N/A'
        output_size = str(summary['module'][key]['output_size'])
        num_params = 0
        for name in summary['module'][key]['params']:
            num_params += (summary['module'][key]['params'][name]['mask'] > 0).sum().item()
        records.append([module_name, input_size, weight_size, output_size, num_params])
    total_num_param = summary['total_num_param']
    total_space_param = summary['total_space_param']

    table = tabulate(records, headers=headers, tablefmt='github')
    content += table + '\n'
    content += '================================================================\n'
    content += 'Total Number of Parameters: {}\n'.format(total_num_param)
    content += 'Total Space of Parameters (MB): {:.2f}\n'.format(total_space_param)
    makedir_exist_ok('./output')
    content_file = open('./output/summary.md', 'w')
    content_file.write(content)
    content_file.close()
    return content


if __name__ == "__main__":
    main()