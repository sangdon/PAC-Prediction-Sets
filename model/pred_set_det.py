import os, sys
import numpy as np
import warnings

import torch as tc

from .pred_set import PredSet, PredSetReg
from .util import *


class PredSetBox(PredSetReg):
    def __init__(self, mdl, eps=0.0, delta=0.0, n=0, var_min=1e-16):
        super().__init__(mdl=mdl, eps=eps, delta=delta, n=n, var_min=var_min)


    def size_volume(self, x):
        lb, ub = self.set_boxapprox(x)

        width = ub[:, 2] - lb[:, 0]
        height = ub[:, 3] - lb[:, 1]
        return width*height        


