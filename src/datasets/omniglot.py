import anytree
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import IMG_EXTENSIONS
from .utils import download_url, extract_file, make_classes_counts, make_data, make_tree, make_flat_index


class Omniglot(Dataset):
    data_name = 'Omniglot'
    file = [
        ('https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
         '68d2efa1b9178cc56df9314c21c6e718'),
        ('https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip',
         '6b91aef0f799c5bb55b94e3f2daec811')
    ]

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
        img, target = Image.open(self.img[index], mode='r').convert('L'), torch.tensor(self.target[index])
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
        img = make_data(self.raw_folder, IMG_EXTENSIONS)
        classes = set()
        train_img = []
        test_img = []
        train_label = []
        test_label = []
        for i in range(len(img)):
            img_i = img[i]
            class_i = '/'.join(os.path.normpath(img_i).split(os.path.sep)[-3:-1])
            classes.add(class_i)
            idx_i = int(os.path.splitext(os.path.basename(img_i))[0].split('_')[1])
            if idx_i <= 10:
                train_img.append(img_i)
            else:
                test_img.append(img_i)
        classes = sorted(list(classes))
        classes_to_labels = {'label': anytree.Node('U', index=[])}
        for c in classes:
            make_tree(classes_to_labels['label'], c.split('/'))
        classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        r = anytree.resolver.Resolver()
        for i in range(len(train_img)):
            train_img_i = train_img[i]
            train_class_i = '/'.join(os.path.normpath(train_img_i).split(os.path.sep)[-3:-1])
            node = r.get(classes_to_labels['label'], train_class_i)
            train_label.append(node.flat_index)
        for i in range(len(test_img)):
            test_img_i = test_img[i]
            test_class_i = '/'.join(os.path.normpath(test_img_i).split(os.path.sep)[-3:-1])
            node = r.get(classes_to_labels['label'], test_class_i)
            test_label.append(node.flat_index)
        train_target = {'label': train_label}
        test_target = {'label': test_label}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)