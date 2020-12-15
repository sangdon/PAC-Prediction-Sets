import os, sys

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, n_in, n_out, n_hiddens=500, n_layers=4):
        super().__init__()
        
        models = []
        for i in range(n_layers):
            n = n_in if i == 0 else n_hiddens
            models.append(nn.Linear(n, n_hiddens))
            models.append(nn.ReLU())
            models.append(nn.Dropout(0.5))
        models.append(nn.Linear(n_hiddens if n_hiddens is not None else n_in, n_out))
        self.model = nn.Sequential(*models)
        
        
    def forward(self, x, training=False):
        if training:
            self.model.train()
        else:
            self.model.eval()
        logits = self.model(x)
        if logits.shape[1] == 1:
            probs = F.sigmoid(logits)
        else:
            probs = F.softmax(logits, -1)
        return {'fh': logits, 'ph': probs, 'yh_top': logits.argmax(-1), 'ph_top': probs.max(-1)[0]}


class Linear(FNN):
    def __init__(self, n_in, n_out, n_hiddens=None):
        super().__init__(n_in, n_out, n_hiddens=None, n_layers=0)


class SmallFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=1)

    
class MidFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=2)

        
class BigFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=4)






