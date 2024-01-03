__author__ = 'Titi Wei'
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as TF
from torch.nn import functional as F
import numpy as np
import os, glob
import random

import sys
sys.path.append('data')
import utils

class BaseDataset(Dataset):
    def __init__(self):
        transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]
        self.transform = torchvision.transforms.Compose(transforms)

        self.totensor = torchvision.transforms.ToTensor()

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    def to_one_hot(self, x, num_classes):
        x = torch.from_numpy(x).type(torch.long)
        x = F.one_hot(x, num_classes=num_classes).type(torch.float32)
        x = x.permute(2, 0, 1)
        return x

class LungDataset(BaseDataset):
    def __init__(self, folder, list_path, num_classes=6, train=True):
        super().__init__()
        self.num_classes = num_classes
        self.train = train

        # open list and glob all ct slice paths
        self.List = utils.open_txt(list_path)
        self.paths = []
        for number in self.List:
            path = os.path.join(folder, number)
            ct_paths = glob.glob(os.path.join(path, 'CT', '*.npy'))
            self.paths += ct_paths

    def __getitem__(self, index):
        ct_path = self.paths[index]
        lung_path = ct_path.replace('CT', 'Lung')
        ct = np.load(ct_path)
        lung = np.load(lung_path)

        # Normalization and transform
        ct = self.normalize(ct)
        ct = self.transform(ct).type(torch.float32)
        # Segmentation class to one hot
        lung = self.to_one_hot(lung, self.num_classes)

        # data augmentation rotate
        if self.train:
            angle = random.choice([0, 90, 180, 270])
            ct = TF.rotate(ct, angle)
            lung = TF.rotate(lung, angle)

        return ct, lung

    def __len__(self):
        return len(self.paths)
    
class TumorDataset(BaseDataset):
    def __init__(self, folder, list_path, num_classes=2, train=True):
        super().__init__()

        self.num_classes = num_classes
        self.train = train

        # the tumor paths that only have tumor
        self.paths = utils.open_txt(list_path)

    def __getitem__(self, index):
        tumor_path = self.paths[index]
        ct_path = tumor_path.replace('Tumor', 'CT')
        lung_path = tumor_path.replace('Tumor', 'Lung')

        ct = np.load(ct_path)
        lung = np.load(lung_path)
        tumor = np.load(tumor_path)

        # Normalization
        ct = self.normalize(ct)

        # lung seg and tumor seg convert to 1 class
        lung = np.where(lung >= 1, 1, 0)
        tumor = np.where(tumor >= 1, 1, 0)

        # only take the part of lung
        union = lung + tumor
        ct = np.where(union >= 1, ct, 0)

        # Tensor transform
        ct = self.transform(ct).type(torch.float32)

        # tumor data convert to one hot
        tumor = self.to_one_hot(tumor, self.num_classes)

        # data augmentation rotate
        if self.train:
            angle = random.choice([0, 90, 180, 270])
            ct = TF.rotate(ct, angle)
            tumor = TF.rotate(tumor, angle)

        return ct, tumor

    def __len__(self):
        return len(self.paths)

    
class TestDataset(BaseDataset):
    def __init__(self, folder, lung_classes=6, tumor_classes=2):
        super().__init__()
        self.lung_classes = lung_classes
        self.tumor_classes = tumor_classes

        self.paths = glob.glob(os.path.join(folder, 'CT', '*.npy'))
        self.paths.sort()

    def __getitem__(self, index):
        ct_path = self.paths[index]
        lung_path = ct_path.replace('CT', 'Lung')
        tumor_path = ct_path.replace('CT', 'Tumor')

        ct = np.load(ct_path)
        lung = np.load(lung_path)
        tumor = np.load(tumor_path)

        tumor = np.where(tumor >= 1, 1, 0)

        # lung and tumor convert to one hot
        lung = self.to_one_hot(lung, self.lung_classes)
        tumor = self.to_one_hot(tumor, self.tumor_classes)

        # Normalization and transform
        ct = self.normalize(ct)
        ct = self.transform(ct).type(torch.float32)

        return ct, lung, tumor

    def __len__(self):
        return len(self.paths)
    
class PredictDataset(BaseDataset):
    def __init__(self, folder):
        super().__init__()

        self.paths = glob.glob(os.path.join(folder, 'CT', '*.npy'))
        self.paths.sort()

    def __getitem__(self, index):
        ct_path = self.paths[index]
        ct = np.load(ct_path)

        # Normalization and transform
        ct = self.normalize(ct)
        ct = self.transform(ct).type(torch.float32)

        return ct
    
    def __len__(self):
        return len(self.paths)
    
if __name__ == '__main__':
    # dataset = CTDataset('/root/VGHTC/No_IV_197_preprocessed', 'trainList.txt', num_classes=72, train=True)
    # dataset = PredictDataset('/root/VGHTC/No_IV_197_preprocessed/000074623G')
    dataset = TumorDataset('/root/VGHTC/No_IV_197_preprocessed', 'tumor_trainList.txt', num_classes=2, train=True)
    print(dataset.__len__())
    for ct in dataset:
        print(ct.shape)
        break