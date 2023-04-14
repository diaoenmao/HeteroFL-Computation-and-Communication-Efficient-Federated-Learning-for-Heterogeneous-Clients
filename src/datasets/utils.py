import anytree
import hashlib
import os
import glob
import gzip
import tarfile
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
from utils import makedir_exist_ok
from .transforms import *

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    classes_to_labels = {classes[i]: i for i in range(len(classes))}
    return classes_to_labels


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_classes_counts(label):
    label = np.array(label)
    if label.ndim > 1:
        label = label.sum(axis=tuple([i for i in range(1, label.ndim)]))
    classes_counts = Counter(label)
    return classes_counts


def make_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(path, md5, **kwargs):
    return md5 == calculate_md5(path, **kwargs)


def check_integrity(path, md5=None):
    if not os.path.isfile(path):
        return False
    if md5 is None:
        return True
    return check_md5(path, md5)


def download_url(url, root, filename, md5):
    from six.moves import urllib
    path = os.path.join(root, filename)
    makedir_exist_ok(root)
    if os.path.isfile(path) and check_integrity(path, md5):
        print('Using downloaded and verified file: ' + path)
    else:
        try:
            print('Downloading ' + url + ' to ' + path)
            urllib.request.urlretrieve(url, path, reporthook=make_bar_updater(tqdm(unit='B', unit_scale=True)))
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + path)
                urllib.request.urlretrieve(url, path, reporthook=make_bar_updater(tqdm(unit='B', unit_scale=True)))
        if not check_integrity(path, md5):
            raise RuntimeError('Not valid downloaded file')
    return


def extract_file(src, dest=None, delete=False):
    print('Extracting {}'.format(src))
    dest = os.path.dirname(src) if dest is None else dest
    filename = os.path.basename(src)
    if filename.endswith('.zip'):
        with zipfile.ZipFile(src, "r") as zip_f:
            zip_f.extractall(dest)
    elif filename.endswith('.tar'):
        with tarfile.open(src) as tar_f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_f, dest)
    elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
        with tarfile.open(src, 'r:gz') as tar_f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_f, dest)
    elif filename.endswith('.gz'):
        with open(src.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(src) as zip_f:
            out_f.write(zip_f.read())
    if delete:
        os.remove(src)
    return


def make_data(root, extensions):
    path = []
    files = glob.glob('{}/**/*'.format(root), recursive=True)
    for file in files:
        if has_file_allowed_extension(file, extensions):
            path.append(os.path.normpath(file))
    return path


def make_img(path, classes_to_labels, extensions=IMG_EXTENSIONS):
    img, label = [], []
    classes = []
    leaf_nodes = classes_to_labels.leaves
    for node in leaf_nodes:
        classes.append(node.name)
    for c in sorted(classes):
        d = os.path.join(path, c)
        if not os.path.isdir(d):
            continue
        for root, _, filenames in sorted(os.walk(d)):
            for filename in sorted(filenames):
                if has_file_allowed_extension(filename, extensions):
                    cur_path = os.path.join(root, filename)
                    img.append(cur_path)
                    label.append(anytree.find_by_attr(classes_to_labels, c).flat_index)
    return img, label


def make_tree(root, name, attribute=None):
    if len(name) == 0:
        return
    if attribute is None:
        attribute = {}
    this_name = name[0]
    next_name = name[1:]
    this_attribute = {k: attribute[k][0] for k in attribute}
    next_attribute = {k: attribute[k][1:] for k in attribute}
    this_node = anytree.find_by_attr(root, this_name)
    this_index = root.index + [len(root.children)]
    if this_node is None:
        this_node = anytree.Node(this_name, parent=root, index=this_index, **this_attribute)
    make_tree(this_node, next_name, next_attribute)
    return


def make_flat_index(root, given=None):
    if given:
        classes_size = 0
        for node in anytree.PreOrderIter(root):
            if len(node.children) == 0:
                node.flat_index = given.index(node.name)
                classes_size = given.index(node.name) + 1 if given.index(node.name) + 1 > classes_size else classes_size
    else:
        classes_size = 0
        for node in anytree.PreOrderIter(root):
            if len(node.children) == 0:
                node.flat_index = classes_size
                classes_size += 1
    return classes_size


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            if isinstance(t, CustomTransform):
                input['img'] = t(input)
            else:
                input['img'] = t(input['img'])
        return input

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
