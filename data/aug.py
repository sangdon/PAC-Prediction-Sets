import sys, os
import torch as tc

class GaussianNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, img):
        img = img + tc.normal(0, self.std, size=img.shape)
        return img

    
class IntensityScaling:
    def __init__(self, min_scale=1.0, max_scale=1.0):
        assert(min_scale <= max_scale)
        self.min_scale = min_scale
        self.max_scale = max_scale
        

    def __call__(self, img):
        s = tc.rand(1)*(self.max_scale-self.min_scale) + self.min_scale
        img = img * s
        return img


class Clamp:
    def __init__(self, mn=0.0, mx=1.0):
        self.mn = mn
        self.mx = mx

    def __call__(self, img):
        img = tc.clamp(img, self.mn, self.mx)
        return img
    
    
def get_aug_tforms(aug_names):
    if aug_names is None:
        return []
    aug_tforms = []
    for aug_name in aug_names:
        if 'noise' in aug_name:
            std = float(aug_name.split(":")[1])
            aug_tforms += [GaussianNoise(std)]
        elif 'intensityscaling' in aug_name:
            min_scale = float(aug_name.split(":")[1])
            max_scale = float(aug_name.split(":")[2])
            aug_tforms += [IntensityScaling(min_scale, max_scale)]
        elif 'clamp' in aug_name:
            mn = float(aug_name.split(":")[1])
            mx = float(aug_name.split(":")[2])
            aug_tforms += [Clamp(mn, mx)]
            
        elif aug_name == 'svhnspecific':
            raise NotImplementedError
            aug_tforms += [
                GaussianNoise(0.1),
            ]
        else:
            raise NotImplementedError

    return aug_tforms
