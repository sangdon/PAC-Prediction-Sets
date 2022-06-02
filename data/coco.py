import os, sys
import numpy as np
import warnings

import torch as tc
from torchvision import transforms as tforms

import data
import data.custom_transforms as ctforms

#IMAGE_SIZE=28

def collate_fn(batch):
    return tuple(zip(*batch))


class COCO(data.DetectionData):
    def __init__(
            self, root, batch_size,
            #image_size=IMAGE_SIZE, color=False,
            train_rnd=True, val_rnd=True, test_rnd=False,
            train_aug=False, val_aug=False, test_aug=False,
            aug_types=[],
            split_ratio={'train': None, 'val': None, 'test': None},
            sample_size={'train': None, 'val': None, 'test': None},
            domain_label=None,
            seed=0,
            num_workers=0,
            year='2017',
            target_labels=None
    ):
        ## default tforms
        tforms_dft = [
            #tforms.Grayscale(3 if color else 1),
            #tforms.Resize([image_size, image_size]) if image_size!=IMAGE_SIZE else ctforms.Identity(),
            tforms.ToTensor(),
        ]

        ## load annotations
        from pycocotools.coco import COCO
        train_anno = COCO(os.path.join(root, 'annotations', 'instances_train'+year+'.json'))
        val_anno = COCO(os.path.join(root, 'annotations', 'instances_val'+year+'.json'))

        # cats = train_anno.loadCats(train_anno.getCatIds())
        # print(cats, len(cats))
        
        ## extract image filenames and labels
        def load_labeled_examples(root, coco, target_labels):            
            img_fns, labels = [], []
            for img_id in list(sorted(coco.imgs.keys())):
                ann_ids = coco.getAnnIds(imgIds=img_id)
                target = coco.loadAnns(ann_ids)
                if len(target) == 0:
                    continue
                target = [i for i in target if i['bbox'][2] > 0 and i['bbox'][3] > 0]
                target_new = {
                    'boxes': data.xywh2xyxy(tc.tensor([i['bbox'] for i in target])),
                    'labels': tc.tensor([i['category_id'] for i in target]),
                    'image_id': tc.tensor([target[0]['image_id']]),
                    'area': tc.tensor([i['area'] for i in target]),
                    'iscrowd': tc.tensor([i['iscrowd'] for i in target]),
                }

                idx_val = tc.zeros_like(target_new['labels']).bool()
                for l in target_labels:
                    idx_val = idx_val | (target_new['labels'] == l)
                target_new['boxes'] = target_new['boxes'][idx_val]
                target_new['labels'] = target_new['labels'][idx_val]
                target_new['area'] = target_new['area'][idx_val]
                target_new['iscrowd'] = target_new['iscrowd'][idx_val]
                
                img_path = coco.loadImgs(img_id)[0]['file_name']
                img_fns.append(os.path.join(root, img_path))
                labels.append(target_new)
            return img_fns, labels

        train_img_fns, train_labels = load_labeled_examples(os.path.join(root, 'train'), train_anno, target_labels)
        val_img_fns, val_labels = load_labeled_examples(os.path.join(root, 'val'), val_anno, target_labels)

        ## shuffle
        train_img_fns, train_labels = data.shuffle_list(train_img_fns, seed), data.shuffle_list(train_labels, seed)
        val_img_fns, val_labels = data.shuffle_list(val_img_fns, seed), data.shuffle_list(val_labels, seed)

        ## split
        assert(split_ratio['train'] is None)
        warnings.warn('I think data.split_list is legacy')
        val_img_fns, test_img_fns = data.split_list([split_ratio['val'], split_ratio['test']], val_img_fns)
        val_labels, test_labels = data.split_list([split_ratio['val'], split_ratio['test']], val_labels)

        data_split = {'train': {'fn': train_img_fns, 'label': train_labels},
                      'val': {'fn': val_img_fns, 'label': val_labels},
                      'test': {'fn': test_img_fns, 'label': test_labels}}
        del train_anno, val_anno
        
        super().__init__(
            root=root, batch_size=batch_size,
            dataset_fn=data.DetectionListDataset,
            data_split=data_split,
            #split_ratio=split_ratio,
            sample_size=sample_size,
            domain_label=domain_label,
            train_rnd=train_rnd, val_rnd=val_rnd, test_rnd=test_rnd,
            train_aug=train_aug, val_aug=val_aug, test_aug=test_aug,
            aug_types=aug_types,
            num_workers=num_workers,
            tforms_dft=tforms_dft, tforms_dft_rnd=tforms_dft,
            collate_fn=collate_fn,
            #ext='png',
            #seed=seed,
        )
        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')
        
            
if __name__ == '__main__':
    dsld = data.COCO('data/coco', 1, sample_size={'train': None, 'val': None, 'test': None}, split_ratio={'train': None, 'val': 0.5, 'test': 0.5})
    print("#train = ", data.compute_num_exs(dsld.train))
    print("#val = ", data.compute_num_exs(dsld.val))
    print("#test = ", data.compute_num_exs(dsld.test))

"""
#train =  50000
#val =  10000
#test =  10000
"""
