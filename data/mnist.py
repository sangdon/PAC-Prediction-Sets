import os, sys
import glob
import time
import random
import pickle
import numpy as np
import types
from PIL import Image

import torch as tc
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

import torchvision.transforms as tforms
import data
import data.custom_transforms as ctforms


class MNISTDataset:
    def __init__(self, x, y, classes, class_to_idx, transform):
        self.x = x
        self.y = y
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform


    def __len__(self):
        return len(self.y)

    
    def __getitem__(self, index):
        sample, target = self.x[index], self.y[index]
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample, target = self.transform((sample, target))
        return sample, target
        

class MNIST:
    def __init__(self, args):

        root = os.path.join('data', args.src.lower())
        
        ## default transforms
        tforms_dft = [
            ctforms.Grayscale(3),
            ctforms.ToTensor(),
        ]
        

        ## transformations for each data split
        tforms_train = tforms_dft
        tforms_val = tforms_dft
        tforms_test = tforms_dft
        print("[tforms_train] ", tforms_train)
        print("[tforms_val] ", tforms_val)
        print("[tforms_test] ", tforms_test)


        ## load data using pytorch datasets
        train_ds = datasets.MNIST(root=root, train=True, download=True, transform=None)
        test_ds = datasets.MNIST(root=root, train=False, download=True, transform=None)

        ## get splits
        x_test, y_test = np.array(test_ds.data), np.array(test_ds.targets)
        
        index_rnd = data.get_random_index(len(x_test), len(x_test), args.seed)
        index_val = index_rnd[:len(index_rnd)//2] # split by half
        index_test = index_rnd[len(index_rnd)//2:]

        x_train, y_train = np.array(train_ds.data), np.array(train_ds.targets)
        x_val, y_val = x_test[index_val], y_test[index_val]
        x_test, y_test = x_test[index_test], y_test[index_test]
               
        ## get class name
        classes, class_to_idx = train_ds.classes, train_ds.class_to_idx

        ## create a data loader for training
        ds = Subset(MNISTDataset(x_train, y_train, classes, class_to_idx, transform=tforms.Compose(tforms_train)),
                    data.get_random_index(len(y_train), len(y_train), args.seed))
        self.train = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

        ## create a data loader for validation
        ds = Subset(MNISTDataset(x_val, y_val, classes, class_to_idx, transform=tforms.Compose(tforms_val)),
                    data.get_random_index(len(y_val), len(y_val), args.seed))
        self.val = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

        ## create a data loader for test
        ds = Subset(MNISTDataset(x_test, y_test, classes, class_to_idx, transform=tforms.Compose(tforms_test)),
                    data.get_random_index(len(y_test), len(y_test), args.seed))
        self.test = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
        
        ## print data statistics
        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')

        
if __name__ == '__main__':
    dsld = data.MNIST(types.SimpleNamespace(src='MNIST', batch_size=100, seed=0, n_workers=10))

    
## MNIST
#train =  60000
#val =  5000
#test =  5000



