import anytree
import codecs
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index


class MNIST(Dataset):
    data_name = 'MNIST'
    file = [('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')]

    def __init__(self, root, split, subset, transform=None):
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
        img, target = Image.fromarray(self.img[index], mode='L'), torch.tensor(self.target[index])
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
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_img = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
        test_img = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
        train_label = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        test_label = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        train_target, test_target = {'label': train_label}, {'label': test_label}
        classes_to_labels = {'label': anytree.Node('U', index=[])}
        classes = list(map(str, list(range(10))))
        for c in classes:
            make_tree(classes_to_labels['label'], [c])
        classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)


class EMNIST(MNIST):
    data_name = 'EMNIST'
    file = [('http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip', '58c8d27c78d21e728a6bc7b3cc06412e')]

    def __init__(self, root, split, subset, transform=None):
        super().__init__(root, split, subset, transform)
        self.img = self.img[self.subset]

    def make_data(self):
        gzip_folder = os.path.join(self.raw_folder, 'gzip')
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith('.gz'):
                extract_file(os.path.join(gzip_folder, gzip_file))
        subsets = ['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist']
        train_img, test_img, train_target, test_target = {}, {}, {}, {}
        digits_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        upper_letters_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                                 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        lower_letters_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                                 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        merged_classes = ['c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z']
        unmerged_classes = list(set(lower_letters_classes) - set(merged_classes))
        classes = {'byclass': digits_classes + upper_letters_classes + lower_letters_classes,
                   'bymerge': digits_classes + upper_letters_classes + unmerged_classes,
                   'balanced': digits_classes + upper_letters_classes + unmerged_classes,
                   'letters': upper_letters_classes + unmerged_classes, 'digits': digits_classes,
                   'mnist': digits_classes}
        classes_to_labels = {s: anytree.Node('U', index=[]) for s in subsets}
        classes_size = {}
        for subset in subsets:
            train_img[subset] = read_image_file(
                os.path.join(gzip_folder, 'emnist-{}-train-images-idx3-ubyte'.format(subset)))
            train_img[subset] = np.transpose(train_img[subset], [0, 2, 1])
            test_img[subset] = read_image_file(
                os.path.join(gzip_folder, 'emnist-{}-test-images-idx3-ubyte'.format(subset)))
            test_img[subset] = np.transpose(test_img[subset], [0, 2, 1])
            train_target[subset] = read_label_file(
                os.path.join(gzip_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(subset)))
            test_target[subset] = read_label_file(
                os.path.join(gzip_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(subset)))
            for c in classes[subset]:
                make_tree(classes_to_labels[subset], c)
            classes_size[subset] = make_flat_index(classes_to_labels[subset])
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)


class FashionMNIST(MNIST):
    data_name = 'FashionMNIST'
    file = [('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
             '8d4fb7e6c68d591d4c3dfef9ec88bf0d'),
            ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
             'bef4ecab320f06d8554ea6380940ec79'),
            ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
             '25c81989df183df01b3e8a0aad5dffbe'),
            ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
             'bb300cfdad3c16e7a12a480ee83cd310')]

    def make_data(self):
        train_img = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
        test_img = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
        train_label = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        test_label = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        train_target = {'label': train_label}
        test_target = {'label': test_label}
        classes_to_labels = {'label': anytree.Node('U', index=[])}
        classes = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']
        for c in classes:
            make_tree(classes_to_labels['label'], c)
        classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape((length, num_rows, num_cols))
        return parsed


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(length).astype(np.int64)
        return parsed