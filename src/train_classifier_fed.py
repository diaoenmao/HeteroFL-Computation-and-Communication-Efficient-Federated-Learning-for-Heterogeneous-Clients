import argparse
import copy
import datetime
import math
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from metrics import Metric
from utils import save, load, to_device, process_control, process_dataset, collate
from logger import Logger

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
cfg['pivot_metric'] = 'Loss'
cfg['pivot'] = float('inf')
cfg['metric_name'] = {'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']}
cfg['batch_size'] = {'train': 16, 'test': 512}


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset['train'])
    data_split = split_dataset(dataset['train'], cfg['num_users'], cfg['split'])
    data_loader = make_data_loader(dataset)
    model = eval('models.{}("global").to(cfg["device"])'.format(cfg['model_name']))
    global_parameters = model.state_dict()
    if cfg['resume_mode'] == 1:
        last_epoch, federation, model, logger = resume(model, cfg['model_tag'])
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, model, _, _ = resume(model, cfg['model_tag'])
        federation = Federation(global_parameters, cfg['rate'])
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        federation = Federation(global_parameters, cfg['rate'])
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)
        train(dataset['train'], data_split, federation, model, logger, epoch)
        test(data_loader['test'], model, logger, epoch)
        logger.safe(False)
        model_state_dict = model.state_dict()
        save_result = {
            'config': cfg, 'epoch': epoch + 1, 'federation': federation, 'model_dict': model_state_dict,
            'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] > logger.tracker['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.tracker['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(dataset, data_split, federation, global_model, logger, epoch):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    idx_user = np.random.choice(range(cfg['num_users']), num_active_users, replace=False)
    local_parameters = federation.distribute(num_active_users)
    for m in range(num_active_users):
        local = Local(dataset, data_split[idx_user[m]])
        local_model = eval('models.{}("local").to(cfg["device"])'.format(cfg['model_name']))
        local_model.load_state_dict(local_parameters[m])
        local.train(local_model, logger, epoch, idx_user[m])
        local_parameters[m] = copy.deepcopy(local_model.state_dict())
    federation.combine(local_parameters)
    return


def test(data_loader, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['img'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test'])
    return


def make_optimizer(model):
    if cfg['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                                  weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if cfg['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'],
                                              gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'],
                                                   gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs']['local'])
    elif cfg['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['factor'],
                                                         patience=cfg['patience'], verbose=True,
                                                         threshold=cfg['threshold'], threshold_mode='rel',
                                                         min_lr=cfg['min_lr'])
    elif cfg['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg['lr'], max_lr=10 * cfg['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def resume(model, model_tag, load_tag='checkpoint', verbose=True):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        last_epoch = checkpoint['epoch']
        federation = checkpoint['federation']
        model.load_state_dict(checkpoint['model_dict'])
        logger = checkpoint['logger']
        if verbose:
            print('Resume from {}'.format(last_epoch))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        federation = Federation(model.state_dict(), cfg['rate'])
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
    return last_epoch, federation, model, logger


class Local:
    def __init__(self, dataset, idx):
        self.data_loader = make_data_loader({'train': SplitDataset(dataset, idx)})['train']

    def train(self, model, logger, epoch, id):
        metric = Metric()
        start_time = time.time()
        model.train()
        optimizer = make_optimizer(model)
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, input in enumerate(self.data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                optimizer.step()
                if (i + 1) % math.ceil((len(self.data_loader) * cfg['log_interval'])) == 0:
                    batch_time = time.time() - start_time
                    lr = optimizer.param_groups[0]['lr']
                    local_epoch_finished_time = datetime.timedelta(
                        seconds=round(batch_time * (len(self.data_loader) - i - 1)))
                    local_finished_time = local_epoch_finished_time + datetime.timedelta(
                        seconds=round((cfg['num_epochs']['local'] - local_epoch) * batch_time * len(self.data_loader)))
                    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                                     'ID: {}, Local/Global Epoch: {}/{}({:.0f}%)'.format(
                                         id, local_epoch, epoch, 100. * (i + 1) / len(self.data_loader)),
                                     'Learning rate: {}'.format(lr), 'Local Epoch Finished Time: {}'.format(
                            local_epoch_finished_time), 'Local Finished Time: {}'.format(local_finished_time)]}
                    logger.append(info, 'train', mean=False)
                    evaluation = metric.evaluate(cfg['metric_name']['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    logger.write('train', cfg['metric_name']['train'])
        return


class Federation:
    def __init__(self, global_parameters, rate):
        self.global_parameters = global_parameters
        self.rate = rate

    def distribute(self, num_active_users):
        idx_i = [None for _ in range(num_active_users)]
        idx = [OrderedDict() for _ in range(num_active_users)]
        local_parameters = [OrderedDict() for _ in range(num_active_users)]
        output_weight = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(num_active_users):
                if parameter_type in ['weight', 'bias', 'running_mean', 'running_var']:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if idx_i[m] is None:
                                idx_i[m] = torch.arange(input_size, device=v.device)
                            input_idx_i_m = idx_i[m]
                            if k == output_weight:
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                local_output_size = int(np.ceil(output_size * self.rate))
                                output_idx_i_m = torch.randperm(output_size, device=v.device)[:local_output_size]
                            idx[m][k] = torch.meshgrid(output_idx_i_m, input_idx_i_m)
                            local_parameters[m][k] = copy.deepcopy(v[idx[m][k]])
                            idx_i[m] = output_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                            local_parameters[m][k] = copy.deepcopy(v[idx[m][k]])
                    else:
                        input_idx_i_m = idx_i[m]
                        idx[m][k] = input_idx_i_m
                        local_parameters[m][k] = copy.deepcopy(v[idx[m][k]])

                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        self.idx = idx
        return local_parameters

    def combine(self, local_parameters):
        idx = self.idx
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            tmp_v, count = v.new_zeros(v.size(), dtype=torch.float32), v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(local_parameters)):
                if parameter_type in ['weight', 'bias', 'running_mean', 'running_var']:
                    tmp_v[idx[m][k]] += local_parameters[m][k]
                    count[idx[m][k]] += 1
                else:
                    tmp_v += local_parameters[m][k]
                    count += 1
            tmp_v[count > 0] = tmp_v[count > 0].div_(count[count > 0])
            v[count > 0] = tmp_v[count > 0].to(v.dtype)
        return


if __name__ == "__main__":
    main()