import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, save, load
from .utils import find_classes, make_img, make_classes_counts


class ImageFolder(Dataset):

    def __init__(self, root, split, subset, transform=None):
        self.data_name = os.path.basename(root)
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.target = self.target[self.subset]
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.classes_size = load(os.path.join(self.processed_folder, 'meta.pt'))
        self.classes_to_labels, self.classes_size = self.classes_to_labels[self.subset], self.classes_size[self.subset]

    def __getitem__(self, index):
        img, target = Image.open(self.img[index], mode='r').convert('RGB'), torch.tensor(self.target[index])
        input = {'img': img, self.subset: target}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.img)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            raise RuntimeError('Dataset not found')
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        classes_to_labels, classes_size = find_classes(os.path.join(self.raw_folder, 'train'))
        train_img, train_label = make_img(os.path.join(self.raw_folder, 'train'), classes_to_labels['label'])
        test_img, test_label = make_img(os.path.join(self.raw_folder, 'test'), classes_to_labels['label'])
        train_target, test_target = {'label': train_label}, {'label': test_label}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)