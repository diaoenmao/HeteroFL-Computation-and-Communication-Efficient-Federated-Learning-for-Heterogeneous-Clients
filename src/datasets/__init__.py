from .mnist import MNIST, EMNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .omniglot import Omniglot
from .imagenet import ImageNet
from .utils import *
from .transforms import *

__all__ = ('MNIST', 'EMNIST', 'FashionMNIST',
           'CIFAR10', 'CIFAR100',
           'Omniglot',
           'ImageNet')