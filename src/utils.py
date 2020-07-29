import collections.abc as container_abcs
import errno
import numpy as np
import os
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import cfg


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, protocol=2, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path, pickle_protocol=protocol)
    elif mode == 'numpy':
        np.save(path, input, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'numpy':
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    cfg['classes_size'] = dataset.classes_size
    return


def process_control():
    cfg['optimizer_name'] = cfg['control']['optimizer_name']
    if cfg['optimizer_name'] == 'SGD':
        cfg['lr'] = 1e-1
        cfg['momentum'] = 0.9
        cfg['weight_decay'] = 5e-4
        cfg['scheduler_name'] = 'MultiStepLR'
        cfg['milestones'] = [100, 200]
        cfg['factor'] = 0.1
        cfg['num_epochs'] = 300
    elif cfg['optimizer_name'] == 'Adam':
        cfg['lr'] = 3e-4
        cfg['betas'] = (0.9, 0.999)
        cfg['weight_decay'] = 5e-4
        cfg['scheduler_name'] = 'ReduceLROnPlateau'
        cfg['factor'] = 0.5
        cfg['min_lr'] = 1e-5
        cfg['num_epochs'] = 200
    else:
        raise ValueError('Not valid optimizer')
    cfg['num_users'] = int(cfg['control']['num_users'])
    cfg['frac'] = float(cfg['control']['frac'])
    cfg['data_split_mode'] = cfg['control']['data_split_mode']
    cfg['model_split_mode'] = cfg['control']['model_split_mode']
    mode_split_rate = {'a': 1, 'b': 0.5, 'c': 0.25, 'd': 0.125, 'e': 0.0625}
    cfg['rate'] = []
    for m in cfg['model_split_mode']:
        cfg['rate'].append(mode_split_rate[m])
    cfg['rate'] = np.repeat(cfg['rate'], cfg['num_users'] // len(cfg['rate'])).tolist()
    cfg['rate'] = cfg['rate'] + [cfg['rate'][-1] for _ in range(cfg['num_users'] - len(cfg['rate']))]
    cfg[cfg['model_name']] = {}
    if cfg['data_split_mode'] != 'none':
        if cfg['data_name'] in ['MNIST', 'FashionMNIST', 'Omniglot']:
            cfg['data_shape'] = [1, 28, 28]
            cfg['num_epochs'] = {'global': 200, 'local': 10}
            cfg['batch_size'] = {'train': 16, 'test': 128}
            if cfg['model_name'] == 'mlp':
                cfg[cfg['model_name']]['hidden_size'] = [512, 256, 128]
            elif cfg['model_name'] == 'conv':
                cfg[cfg['model_name']]['hidden_size'] = [64, 128, 256, 512]
            else:
                raise ValueError('Not valid model name')
        elif cfg['data_name'] in ['SVHN', 'CIFAR10', 'CIFAR100']:
            cfg['data_shape'] = [3, 32, 32]
            cfg['num_epochs'] = {'global': 200, 'local': 10}
            cfg['batch_size'] = {'train': 16, 'test': 128}
            if cfg['model_name'] == 'mlp':
                cfg[cfg['model_name']]['hidden_size'] = [512, 256, 128]
            elif cfg['model_name'] == 'conv':
                cfg[cfg['model_name']]['hidden_size'] = [64, 128, 256, 512]
            else:
                raise ValueError('Not valid model name')
        elif cfg['data_name'] in ['ImageNet']:
            cfg['data_shape'] = [3, 224, 224]
            if cfg['model_name'] == 'mlp':
                raise NotImplementedError
            elif cfg['model_name'] == 'conv':
                raise NotImplementedError
            else:
                raise ValueError('Not valid model name')
        else:
            raise ValueError('Not valid dataset')
    else:
        if cfg['data_name'] in ['MNIST', 'FashionMNIST', 'Omniglot']:
            cfg['data_shape'] = [1, 28, 28]
            cfg['num_epochs'] = {'global': 200}
            cfg['batch_size'] = {'train': 128, 'test': 256}
            if cfg['model_name'] == 'mlp':
                cfg[cfg['model_name']]['hidden_size'] = [512, 256, 128]
            elif cfg['model_name'] == 'conv':
                cfg[cfg['model_name']]['hidden_size'] = [64, 128, 256, 512]
            else:
                raise ValueError('Not valid model name')
        elif cfg['data_name'] in ['SVHN', 'CIFAR10', 'CIFAR100']:
            cfg['data_shape'] = [3, 32, 32]
            cfg['num_epochs'] = {'global': 200}
            cfg['batch_size'] = {'train': 128, 'test': 256}
            if cfg['model_name'] == 'mlp':
                cfg[cfg['model_name']]['hidden_size'] = [512, 256, 128]
            elif cfg['model_name'] == 'conv':
                cfg[cfg['model_name']]['hidden_size'] = [64, 128, 256, 512]
            else:
                raise ValueError('Not valid model name')
        elif cfg['data_name'] in ['ImageNet']:
            cfg['data_shape'] = [3, 224, 224]
            if cfg['model_name'] == 'mlp':
                raise NotImplementedError
            elif cfg['model_name'] == 'conv':
                raise NotImplementedError
            else:
                raise ValueError('Not valid model name')
        else:
            raise ValueError('Not valid dataset')
    return


def make_stats(dataset):
    if os.path.exists('./data/stats/{}.pt'.format(dataset.data_name)):
        stats = load('./data/stats/{}.pt'.format(dataset.data_name))
    elif dataset is not None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
        stats = Stats(dim=1)
        with torch.no_grad():
            for input in data_loader:
                stats.update(input['img'])
        save(stats, './data/stats/{}.pt'.format(dataset.data_name))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def make_optimizer(model, lr):
    if cfg['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=cfg['momentum'],
                                  weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if cfg['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs']['global'],
                                                         eta_min=cfg['min_lr'])
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


def resume(model, model_tag, optimizer=None, scheduler=None, load_tag='checkpoint', verbose=True):
    if cfg['data_split_mode'] != 'none':
        if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
            checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
            last_epoch = checkpoint['epoch']
            data_split = checkpoint['data_split']
            federation = checkpoint['federation']
            model.load_state_dict(checkpoint['model_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_dict'])
            logger = checkpoint['logger']
            if verbose:
                print('Resume from {}'.format(last_epoch))
        else:
            print('Not exists model tag: {}, start from scratch'.format(model_tag))
            from datetime import datetime
            from logger import Logger
            last_epoch = 1
            data_split = None
            federation = None
            logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
            logger = Logger(logger_path)
        return last_epoch, data_split, federation, model, optimizer, scheduler, logger
    else:
        if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
            checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_dict'])
            logger = checkpoint['logger']
            if verbose:
                print('Resume from {}'.format(last_epoch))
        else:
            print('Not exists model tag: {}, start from scratch'.format(model_tag))
            from datetime import datetime
            from logger import Logger
            last_epoch = 1
            logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
            logger = Logger(logger_path)
        return last_epoch, model, optimizer, scheduler, logger


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input