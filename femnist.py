import torch
# from torch.utils.data import Dataset
from torchvision.datasets import MNIST
# from PIL import Image
import os.path


class FEMNIST(MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(MNIST, self).__init__(root, transform=None, target_transform=None)
        if train:
            self.data, self.targets = torch.load('train.pt')
        else:
            self.data, self.targets = torch.load('test.pt')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
