import chainer
import numpy as np
from chainer.dataset import dataset_mixin


class CIFAR100(dataset_mixin.DatasetMixin):
    def __init__(self, test=False):
        d_train, d_test = chainer.datasets.get_cifar100(ndim=3, withlabel=True, scale=255)
        if test:
            self.dset = d_test
        else:
            self.dset = d_train

        print("load cifar-100.  size:{}, shape(image): {}".format(len(self),self.dset[0][0].shape))

    def __len__(self):
        return len(self.dset)

    def get_example(self, i):
        image = np.asarray(self.dset[i][0] / 128. - 1., np.float32)
        image += np.random.uniform(size=image.shape, low=0., high=1. / 128)
        return image, self.dset[i][1]

class CIFAR10(dataset_mixin.DatasetMixin):
    def __init__(self, test=False):
        d_train, d_test = chainer.datasets.get_cifar10(ndim=3, withlabel=True, scale=255)
        if test:
            self.dset = d_test
        else:
            self.dset = d_train

        print("load cifar-10.  size:{}, shape(image): {}".format(len(self),self.dset[0][0].shape))

    def __len__(self):
        return len(self.dset)

    def get_example(self, i):
        image = np.asarray(self.dset[i][0] / 128. - 1., np.float32)
        image += np.random.uniform(size=image.shape, low=0., high=1. / 128)
        return image, self.dset[i][1]