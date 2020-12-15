import os, sys
import time
#import numpy as np

#import torch as tc
#import torch.tensor as T
#import torch.nn as nn

# sys.path.append("../")
# from .calibrator import BaseCalibrator
# from classification.utils import *

from learning import *
from uncertainty import *


class TempScalingLearner(ClsLearner):
    def __init__(self, mdl, params=None, name_postfix='cal_temp'):
        super().__init__(mdl, params, name_postfix)
        self.loss_fn_train = loss_xe
        self.loss_fn_val = loss_xe
        self.loss_fn_test = loss_01
        self.T_min = 1e-9
        
        
    def _train_epoch_batch_end(self, i_epoch):
        [T.data.clamp_(self.T_min) for T in self.mdl.parameters()]


    def test(self, ld, mdl=None, loss_fn=None, ld_name=None, verbose=False):
        t_start = time.time()
        
        ## compute classification error
        error, ece, *_ = super().test(ld, mdl, loss_fn)

        ## compute confidence distributions
        ph_list = compute_top_conf(self.mdl if mdl is None else mdl, ld, self.params.device)
        mn = ph_list.min()
        Q1 = ph_list.kthvalue(int(round(ph_list.size(0)*0.25)))[0]
        Q2 = ph_list.median()
        Q3 = ph_list.kthvalue(int(round(ph_list.size(0)*0.75)))[0]
        mx = ph_list.max()
        av = ph_list.mean()
        ph_dist = {'min': mn, 'Q1': Q1, 'Q2': Q2, 'Q3': Q3, 'max': mx, 'mean': av}

        if verbose:
            print('[test%s, %f secs.] test error = %.2f%%, ece = %.2f%%'%(
                ': %s'%(ld_name if ld_name else ''), time.time()-t_start, error*100.0, ece*100.0))
            print(
                f'[ph distribution] '
                f'min = {mn:.4f}, 1st-Q = {Q1:.4f}, median = {Q2:.4f}, 3rd-Q = {Q3:.4f}, max = {mx:.4f}, mean = {av:.4f}'
            )

        return error, ece, ph_list

