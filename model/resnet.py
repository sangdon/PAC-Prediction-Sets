import os, sys

import torch as tc
from torch import nn
import torch.nn.functional as F
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, n_labels, resnet_id, pretrained=False):
        super().__init__()
        self.model = getattr(models, 'resnet%d'%(resnet_id))(num_classes=n_labels, pretrained=pretrained)
        def feat_hook(model, input, output):
            self.feat = tc.flatten(output, 1)
        self.model.avgpool.register_forward_hook(feat_hook)
        

    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()
        x = self.model(x)        
        return {'fh': x, 'ph': F.softmax(x, -1), 'yh_top': x.argmax(-1), 'ph_top': F.softmax(x, -1).max(-1)[0], 'feat': self.feat}
    

def ResNet18(n_labels, pretrained=False):
    return ResNet(n_labels, 18, pretrained=pretrained)


def ResNet152(n_labels, pretrained=False):
    return ResNet(n_labels, 152, pretrained=pretrained)


