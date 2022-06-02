import os, sys
import numpy as np
import time
import glob
import random
import math
#from typing import Any, Callable, cast, Dict, List, Optional, Tuple
#from PIL import Image
import pickle
import warnings

import torch as tc
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def decode_input(x):
    if type(x) is tuple:
        ## assume (img, label) tupble
        img = x[0]
        label = x[1]
    else:
        img = x
        label = None
    return img, label




"""
Simple wrapper functions
"""
def compute_num_exs(ld, verbose=False):
    n = 0
    t = time.time()
    for x, _ in ld:
        n += x.shape[0]
        if verbose:
            print("[%f sec.] n = %d"%(time.time()-t, n))
            t = time.time()
    return n

def xywh2xyxy(xywh):
    xyxy = xywh.clone()
    if len(xyxy.size()) == 2:
        xyxy[:, 2:] = xywh[:, :2] + xywh[:, 2:]
    else:
        xyxy[2:] = xywh[:2] + xywh[2:]
    return xyxy


def xyxy2xywh(xyxy):
    xywh = xyxy.clone()
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
    return xywh


"""
functions/classes for data loaders
"""
    
def shuffle_list(list_ori, seed):
    random.seed(seed)
    random.shuffle(list_ori)
    random.seed(int(time.time()))
    return list_ori


def load_data_splits(root, ext, seed):
    
    fns_train = glob.glob(os.path.join(root, 'train', '**', '**.'+ext))
    fns_val = glob.glob(os.path.join(root, 'val', '**', '**.'+ext))
    fns_test = glob.glob(os.path.join(root, 'test', '**', '**.'+ext))
    
    ## shuffle
    fns_train = shuffle_list(fns_train, seed)
    fns_val = shuffle_list(fns_val, seed)
    fns_test = shuffle_list(fns_test, seed)

    return {'train': fns_train, 'val': fns_val, 'test': fns_test}


def get_random_index(n_ori, n, seed):
    random.seed(seed)
    if n_ori < n:
        index = [random.randint(0, n_ori-1) for _ in range(n)]
    else:
        index = [i for i in range(n_ori)]
        random.shuffle(index)
        index = index[:n]
        
    random.seed(time.time())
    return index



def find_classes(root):
    classes = [d.name for s in ['train', 'val', 'test'] for d in os.scandir(os.path.join(root, s)) if d.is_dir()]
    classes = list(set(classes))
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_class_name(fn):
    return fn.split('/')[-2]


def make_dataset(fn_list, class_to_idx):
    instances = []
    for fn in fn_list:
        class_idx = class_to_idx[get_class_name(fn)]
        item = fn, class_idx
        instances.append(item)
    return instances

    
class ImageListDataset:
    def __init__(self, fn_list, classes, class_to_idx, transform=None, loader=default_loader, return_path=False):
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = make_dataset(fn_list, class_to_idx)
        self.return_path = return_path

        
    def __getitem__(self, index):
        path, target = self.samples[index]
        target = target 
        sample = self.loader(path)
        if self.transform is not None:
            sample, target = self.transform((sample, target))
        if self.return_path:
            return path, sample, target
        else:
            return sample, target


    def __len__(self):
        return len(self.samples)



class ConcatDataLoader:
    def __init__(self, ld_list):
        self.ld_list = ld_list
        assert(len(self.ld_list) > 0)
        for ld in self.ld_list:
            assert(len(ld) > 0)
        #self.index = 0


    def __iter__(self):
        self.index = 0
        random.shuffle(self.ld_list)
        self.ld = iter(self.ld_list[self.index])
        return self

    def __next__(self):
        try:
            return next(self.ld)
        except StopIteration:
            self.index = self.index + 1
            if self.index >= len(self.ld_list):
                #self.index = 0
                raise StopIteration
            else:
                self.ld = iter(self.ld_list[self.index])
                return next(self.ld) ## assume ld is not empty since it passed the assertion in __init__()
    

"""
functions for detection
"""
def xywh2xyxy(xywh):
    xyxy = xywh.clone()
    if len(xyxy.size()) == 2:
        xyxy[:, 2:] = xywh[:, :2] + xywh[:, 2:]
    else:
        xyxy[2:] = xywh[:2] + xywh[2:]
    return xyxy


def xyxy2xywh(xyxy):
    xywh = xyxy.clone()
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
    return xywh


def split_list(split_ratio, list_ori):
    list_split = []
    n_start = 0
    for i, ratio in enumerate(split_ratio):
        n = math.floor(len(list_ori)*ratio)
        if i+1 == len(split_ratio):
            list_split.append(list_ori[n_start:])
        else:            
            list_split.append(list_ori[n_start:n_start+n])
        n_start += n
    random.seed(time.time())
    return list_split


class DetectionListDataset:
    def __init__(self, split, transform=None, target_transform=None, loader=default_loader, domain_label=None):
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.samples = [(fn, label) for fn, label in zip(split['fn'], split['label'])]
        self.domain_label = domain_label
        
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        target = target if self.domain_label is None else self.domain_label
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)
    

class DetectionData:
    def __init__(self, root, batch_size,
                 dataset_fn,
                 data_split, 
                 #split_ratio,
                 sample_size,
                 domain_label,
                 train_rnd, val_rnd, test_rnd,
                 train_aug, val_aug, test_aug,
                 aug_types,
                 num_workers,
                 tforms_dft, tforms_dft_rnd,
                 collate_fn=None,
                 #ext,
                 #seed,
    ):
        # ## data augmentation tforms
        # tforms_aug = get_aug_tforms(aug_types)

        ## tforms for each data split
        tforms_train = tforms_dft_rnd if train_rnd else tforms_dft
        tforms_train = tforms_train + tforms_aug if train_aug else tforms_train
        tforms_val = tforms_dft_rnd if val_rnd else tforms_dft
        tforms_val = tforms_val + tforms_aug if val_aug else tforms_val
        tforms_test = tforms_dft_rnd if test_rnd else tforms_dft
        tforms_test = tforms_test + tforms_aug if test_aug else tforms_test

        print("[tforms_train] ", tforms_train)
        print("[tforms_val] ", tforms_val)
        print("[tforms_test] ", tforms_test)
        
        # ## splits
        # split_list = get_split_list(split_ratio, root, ext, seed)
        # classes, class_to_idx = find_classes(root)

        ## truncate samples
        for name, value in data_split.items():
            if sample_size[name] is None:
                continue
            data_split[name] = {k: v[:sample_size[name]] for k, v in value.items()}

        ## create loaders
        dataset = dataset_fn(data_split['train'], transform=transforms.Compose(tforms_train), domain_label=domain_label)
        self.train = DataLoader(dataset, batch_size=batch_size, shuffle=train_rnd, num_workers=num_workers, collate_fn=collate_fn)

        dataset = dataset_fn(data_split['val'], transform=transforms.Compose(tforms_val), domain_label=domain_label)
        self.val = DataLoader(dataset, batch_size=batch_size, shuffle=val_rnd, num_workers=num_workers, collate_fn=collate_fn)

        dataset = dataset_fn(data_split['test'], transform=transforms.Compose(tforms_test), domain_label=domain_label)
        self.test = DataLoader(dataset, batch_size=batch_size, shuffle=test_rnd, num_workers=num_workers, collate_fn=collate_fn)

