import torch
import datasets
import numpy as np
from config import cfg
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


def fetch_dataset(data_name, subset):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name == 'MNIST':
        dataset['train'] = datasets.MNIST(root=root, split='train', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        dataset['test'] = datasets.MNIST(root=root, split='test', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    elif data_name == 'CIFAR10':
        dataset['train'] = datasets.CIFAR10(root=root, split='train', subset=subset, transform=datasets.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        dataset['test'] = datasets.CIFAR10(root=root, split='test', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif data_name in ['PennTreebank', 'WikiText2', 'WikiText103']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\')'.format(data_name))
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def split_dataset(dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'], label_split = iid(dataset['train'], num_users)
        data_split['test'], _ = iid(dataset['test'], num_users)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split['train'], label_split = non_iid(dataset['train'], num_users)
        data_split['test'], _ = non_iid(dataset['test'], num_users, label_split)
    else:
        raise ValueError('Not valid data split mode')
    return data_split, label_split


def iid(dataset, num_users):
    if cfg['data_name'] in ['MNIST', 'CIFAR10']:
        label = torch.tensor(dataset.target)
    elif cfg['data_name'] in ['WikiText2']:
        label = dataset.token
    else:
        raise ValueError('Not valid data name')
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    label_split = {}
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        label_split[i] = torch.unique(label[data_split[i]]).tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split, label_split


def non_iid(dataset, num_users, label_split=None):
    label = np.array(dataset.target)
    cfg['non-iid-n'] = int(cfg['data_split_mode'].split('-')[-1])
    shard_per_user = cfg['non-iid-n']
    data_split = {i: [] for i in range(num_users)}
    label_idx_split = {}
    for i in range(len(label)):
        label_i = label[i].item()
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)
    shard_per_class = int(shard_per_user * num_users / cfg['classes_size'])
    for label_i in label_idx_split:
        label_idx = label_idx_split[label_i]
        num_leftover = len(label_idx) % shard_per_class
        leftover = label_idx[-num_leftover:] if num_leftover > 0 else []
        new_label_idx = np.array(label_idx[:-num_leftover]) if num_leftover > 0 else np.array(label_idx)
        new_label_idx = new_label_idx.reshape((shard_per_class, -1)).tolist()
        for i, leftover_label_idx in enumerate(leftover):
            new_label_idx[i] = np.concatenate([new_label_idx[i], [leftover_label_idx]])
        label_idx_split[label_i] = new_label_idx
    if label_split is None:
        label_split = list(range(cfg['classes_size'])) * shard_per_class
        label_split = torch.tensor(label_split)[torch.randperm(len(label_split))].tolist()
        label_split = np.array(label_split).reshape((num_users, -1)).tolist()
        for i in range(len(label_split)):
            label_split[i] = np.unique(label_split[i]).tolist()
    for i in range(num_users):
        for label_i in label_split[i]:
            idx = torch.arange(len(label_idx_split[label_i]))[torch.randperm(len(label_idx_split[label_i]))[0]].item()
            data_split[i].extend(label_idx_split[label_i].pop(idx))
    return data_split, label_split


def make_data_loader(dataset):
    data_loader = {}
    for k in dataset:
        data_loader[k] = torch.utils.data.DataLoader(dataset=dataset[k], shuffle=cfg['shuffle'][k],
                                                     batch_size=cfg['batch_size'][k], pin_memory=True,
                                                     num_workers=cfg['num_workers'], collate_fn=input_collate)
    return data_loader


class SplitDataset(Dataset):
    def __init__(self, dataset, idx):
        super().__init__()
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        input = self.dataset[self.idx[index]]
        return input


class BatchDataset(Dataset):
    def __init__(self, dataset, seq_length):
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.S = dataset[0]['label'].size(0)
        self.idx = list(range(0, self.S, seq_length))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        seq_length = min(self.seq_length, self.S - index)
        input = {'label': self.dataset[:]['label'][:, self.idx[index]:self.idx[index] + seq_length]}
        return input