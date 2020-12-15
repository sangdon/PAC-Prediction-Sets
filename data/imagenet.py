import os, sys
import glob

import torch as tc
from torchvision import transforms as tforms
from torchvision.datasets.utils import check_integrity

import data
import data.custom_transforms as ctforms


def _load_meta_file(meta_file):
    if check_integrity(meta_file):
        return tc.load(meta_file)
    else:
        raise RuntimeError("Meta file not found or corrupted.")

    
def label_to_name(root, label_to_wnid):
    meta_file = os.path.join(root, 'meta.bin')
    wnid_to_names = _load_meta_file(meta_file)[0]

    names = [wnid_to_names[wnid][0].replace(' ', '_').replace('\'', '_') for wnid in label_to_wnid]
    return names


class ImageNet(data.ImageData):
    def __init__(
            self, root, batch_size,
            image_size=None, color=True,
            train_rnd=True, val_rnd=True, test_rnd=False,
            train_aug=False, val_aug=False, test_aug=False,
            normalize=True,
            aug_types=[],
            num_workers=4,
            domain_label=None,

    ):
        assert(color)
        
        ## default tforms
        tforms_dft = [
            tforms.Resize(256),
            tforms.CenterCrop(224),
            tforms.ToTensor(),
            tforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else ctform.Identity(),
        ]
        tforms_dft_rnd = [
            tforms.RandomResizedCrop(224),
            tforms.RandomHorizontalFlip(),
            tforms.ToTensor(),
            tforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else ctform.Identity(),
        ]

        # ## compute label histogram
        # ##TODO
        # names_val = [n.split('/')[-1] for n in glob.glob(os.path.join(root, 'val', 'n*'))]
        # hist_names = {n: len(glob.glob(os.path.join(root, 'val', n, '*.JPEG'))) for n in names_val}
        # print(hist_names)
        # print(sorted(hist_names.values()))
        # sys.exit()
                  
        super().__init__(
            root=root, batch_size=batch_size,
            #image_size=image_size, color=color,
            domain_label=domain_label,
            train_rnd=train_rnd, val_rnd=val_rnd, test_rnd=test_rnd,
            train_aug=train_aug, val_aug=val_aug, test_aug=test_aug,
            aug_types=aug_types,
            num_workers=num_workers,
            tforms_dft=tforms_dft, tforms_dft_rnd=tforms_dft_rnd,
        )

        ## add id-name map
        self.names = label_to_name(root, self.test.dataset.classes)
        
            
if __name__ == '__main__':
    dsld = data.ImageNet('data/imagenet', 100)
    print("#train = ", data.compute_num_exs(dsld.train, verbose=True))
    print("#val = ", data.compute_num_exs(dsld.val))
    print("#test = ", data.compute_num_exs(dsld.test))

#train =  1281167
#val =  25000
#test =  25000

