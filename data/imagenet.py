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
from torchvision.datasets.folder import pil_loader

sys.path.append('.')
import data
import data.custom_transforms as ctforms


def find_classes(file_list):
    name = [f.split('/')[0] for f in file_list]
    classes = sorted(set(name))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
        

class ImageNetDataset:
    def __init__(self, file_list, class_to_idx, transform):
        self.file_list = file_list
        self.class_to_idx = class_to_idx
        self.transform = transform


    def __len__(self):
        return len(self.file_list)

    
    def __getitem__(self, index):
        file_name = self.file_list[index]
        sample = pil_loader(file_name)
        target = self.class_to_idx[file_name.split('/')[-2]]
        
        if self.transform is not None:
            sample, target = self.transform((sample, target))
        return sample, target
        

class ImageNet:
    def __init__(self, args):

        root = os.path.join('data', args.src.lower())
        root_image = os.path.join(root, 'images')
        
        # default transforms
        tforms_dft = [
            ctforms.Resize(256),
            ctforms.CenterCrop(224),
            ctforms.ToTensor(),
            ctforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]
        

        # transformations for each data split
        tforms_train = tforms_dft # no complex transformation for training
        tforms_val = tforms_dft
        tforms_test = tforms_dft
        print("[tforms_train] ", tforms_train)
        print("[tforms_val] ", tforms_val)
        print("[tforms_test] ", tforms_test)

        

        # load data
        train_files_rel = open(os.path.join(root, 'train.txt')).read().splitlines()
        val_files_rel = open(os.path.join(root, 'val.txt')).read().splitlines()
        test_files_rel = open(os.path.join(root, 'test.txt')).read().splitlines()

        train_files = [os.path.join(root_image, 'train', rel_path) for rel_path in train_files_rel]
        val_files = [os.path.join(root_image, 'val', rel_path) for rel_path in val_files_rel]
        test_files = [os.path.join(root_image, 'test', rel_path) for rel_path in test_files_rel]

        # init meta info
        classes, class_to_idx = find_classes(train_files_rel+val_files_rel+test_files_rel)

        # update sample size
        if not hasattr(args, 'n_train'):
            args.n_train = len(train_files)
        if not hasattr(args, 'n_val'):
            args.n_val = len(val_files)
        if not hasattr(args, 'n_test'):
            args.n_test = len(test_files)
        
        # randomly split data
        train_index_rnd = data.get_random_index(len(train_files), args.n_train, args.seed)
        val_index_rnd = data.get_random_index(len(val_files), args.n_val, args.seed)
        test_index_rnd = data.get_random_index(len(test_files), args.n_test, args.seed)
        
        train_files_rnd = [train_files[i] for i in train_index_rnd]
        val_files_rnd = [val_files[i] for i in val_index_rnd]
        test_files_rnd = [test_files[i] for i in test_index_rnd]

        # init datasetes
        train_ds = ImageNetDataset(train_files_rnd, class_to_idx, tforms.Compose(tforms_train))
        val_ds = ImageNetDataset(val_files_rnd, class_to_idx, tforms.Compose(tforms_val))
        test_ds = ImageNetDataset(test_files_rnd, class_to_idx, tforms.Compose(tforms_test))

        # init data loaders
        self.train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
        self.val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
        self.test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

        ## print data statistics
        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')

        
if __name__ == '__main__':
    dsld = data.ImageNet(types.SimpleNamespace(src='ImageNet', batch_size=100, seed=0, n_workers=10))

    for x, y in dsld.val:
        print(x.shape)
        print(y)
    
## MNIST
#train =  60000
#val =  5000
#test =  5000



