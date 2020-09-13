import torch
import datasets
import numpy as np
from config import cfg
from collections import defaultdict
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
        dataset['valid'] = eval('datasets.{}(root=root, split=\'valid\')'.format(data_name))
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
    label = np.array(dataset.target)
    if data_split_mode == 'iid':
        num_items = round(len(dataset) / num_users)
        data_split, label_split, idx = {}, {}, list(range(len(dataset)))
        for i in range(num_users):
            num_items_i = min(len(idx), num_items)
            data_split[i] = np.random.choice(idx, num_items_i, replace=False).tolist()
            label_split[i] = np.unique(label[data_split[i]]).tolist()
            idx = list(set(idx) - set(data_split[i]))
    elif cfg['data_split_mode'] == 'non-iid':
        num_items = round(len(dataset) / num_users)
        idx_shard = [i for i in range(0, len(dataset), num_items // 2)]
        data_split, label_split, idx = {}, {}, np.arange(len(dataset))
        sorted_indices = np.argsort(label).tolist()
        idx = idx[sorted_indices]
        for i in range(num_users):
            pivot = np.random.choice(idx_shard, 2, replace=False)
            data_split[i] = idx[pivot[0]:(pivot[0] + num_items // 2)].tolist() + \
                            idx[pivot[1]:(pivot[1] + num_items // 2)].tolist()
            label_split[i] = np.unique(label[data_split[i]]).tolist()
            idx_shard = list(set(idx_shard) - set(pivot))
    else:
        raise ValueError('Not valid data split mode')
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