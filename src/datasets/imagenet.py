import anytree
import numpy as np
import os
import shutil
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from utils import check_exists, save, load
from .utils import extract_file, make_classes_counts, make_img, make_tree, make_flat_index

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageNet(Dataset):
    data_name = 'ImageNet'

    def __init__(self, root, split, subset, size, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform = transform
        self.size = size
        if not check_exists(os.path.join(self.processed_folder, str(self.size))):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, str(self.size), '{}.pt'.format(self.split)))
        self.target = self.target[self.subset]
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.classes_size = load(os.path.join(self.processed_folder, str(self.size), 'meta.pt'))
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
        save(train_set, os.path.join(self.processed_folder, str(self.size), 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, str(self.size), 'test.pt'))
        save(meta, os.path.join(self.processed_folder, str(self.size), 'meta.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}\nSize: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset, self.size,
            self.transform.__repr__())
        return fmt_str

    def make_data(self):
        if not check_exists(os.path.join(self.raw_folder, 'base')):
            train_path = os.path.join(self.raw_folder, 'ILSVRC2012_img_train')
            test_path = os.path.join(self.raw_folder, 'ILSVRC2012_img_val')
            meta_path = os.path.join(self.raw_folder, 'ILSVRC2012_devkit_t12')
            extract_file(os.path.join(self.raw_folder, 'ILSVRC2012_img_train.tar'), train_path)
            extract_file(os.path.join(self.raw_folder, 'ILSVRC2012_img_val.tar'), test_path)
            extract_file(os.path.join(self.raw_folder, 'ILSVRC2012_devkit_t12.tar'), meta_path)
            for archive in [os.path.join(train_path, archive) for archive in os.listdir(train_path)]:
                extract_file(archive, os.path.splitext(archive)[0], delete=True)
            classes_to_labels, classes_size = make_meta(meta_path)
            with open(os.path.join(meta_path, 'data', 'ILSVRC2012_validation_ground_truth.txt'), 'r') as f:
                test_id = f.readlines()
            test_id = [int(i) for i in test_id]
            test_img = sorted([os.path.join(test_path, file) for file in os.listdir(test_path)])
            test_wnid = []
            for test_id_i in test_id:
                test_node_i = anytree.find_by_attr(classes_to_labels['label'], name='id', value=test_id_i)
                test_wnid.append(test_node_i.name)
            for test_wnid_i in set(test_wnid):
                os.mkdir(os.path.join(test_path, test_wnid_i))
            for test_wnid_i, test_img in zip(test_wnid, test_img):
                shutil.move(test_img, os.path.join(test_path, test_wnid_i, os.path.basename(test_img)))
            shutil.move(train_path, os.path.join(self.raw_folder, 'base', 'ILSVRC2012_img_train'))
            shutil.move(test_path, os.path.join(self.raw_folder, 'base', 'ILSVRC2012_img_val'))
            shutil.move(meta_path, os.path.join(self.raw_folder, 'base', 'ILSVRC2012_devkit_t12'))
        if not check_exists(os.path.join(self.raw_folder, str(self.size))):
            raise ValueError('Need to run resizer')
        classes_to_labels, classes_size = make_meta(os.path.join(self.raw_folder, 'base', 'ILSVRC2012_devkit_t12'))
        train_img, train_label = make_img(os.path.join(self.raw_folder, str(self.size), 'ILSVRC2012_img_train'),
                                          classes_to_labels['label'])
        test_img, test_label = make_img(os.path.join(self.raw_folder, str(self.size), 'ILSVRC2012_img_val'),
                                        classes_to_labels['label'])
        train_target = {'label': train_label}
        test_target = {'label': test_label}
        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)


def make_meta(path):
    import scipy.io as sio
    meta = sio.loadmat(os.path.join(path, 'data', 'meta.mat'), squeeze_me=True)['synsets']
    num_children = list(zip(*meta))[4]
    leaf_meta = [meta[i] for (i, n) in enumerate(num_children) if n == 0]
    branch_meta = [meta[i] for (i, n) in enumerate(num_children) if n > 0]
    names, attributes = [], []
    for i in range(len(leaf_meta)):
        name, attribute = make_node(leaf_meta[i], branch_meta)
        names.append(name)
        attributes.append(attribute)
    classes_to_labels = {'label': anytree.Node('U', index=[])}
    classes = []
    for (name, attribute) in zip(names, attributes):
        make_tree(classes_to_labels['label'], name, attribute)
        classes.append(name[-1])
    classes_size = {'label': make_flat_index(classes_to_labels['label'], classes)}
    return classes_to_labels, classes_size


def make_node(node, branch):
    id, wnid, classes = node.item()[:3]
    if classes == 'entity':
        name = []
        attribute = {'id': [], 'class': []}
    for i in range(len(branch)):
        branch_children = branch[i].item()[5]
        if (isinstance(branch_children, int) and id == branch_children) or (
                isinstance(branch_children, np.ndarray) and id in branch_children):
            parent_name, parent_attribute = make_node(branch[i], branch)
            name = parent_name + [wnid]
            attribute = {'id': parent_attribute['id'] + [id], 'class': parent_attribute['class'] + [classes]}
            break
    return name, attribute