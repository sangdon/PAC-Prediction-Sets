import os, sys
import time

from learning import *
from learning.util_det import *

class DetLearner(BaseLearner):
    def __init__(self, mdl, params=None, name_postfix=None):
        super().__init__(mdl, params, name_postfix)
        #self.loss_fn_train = loss_xe
        #self.loss_fn_val = loss_01
        #self.loss_fn_test = loss_01

        
    def train(self, ld_tr, ld_val, ld_test=None):
        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return

        raise NotImplementedError
    
        
    def test(self, ld, mdl=None, loss_fn=None, ld_name=None, verbose=False):
        mdl = self.mdl if mdl is None else mdl
        mdl.eval()
        
        evaluate(self.mdl if mdl is None else mdl, ld, self.params.device)
        
        # t_start = time.time()
        # error, *_ = super().test(ld, mdl, loss_fn)
        # ece = compute_ece(self.mdl if mdl is None else mdl, ld, self.params.device)
        
        # if verbose:
        #     print('[test%s, %f secs.] classificaiton error = %.2f%%, calibration error = %.2f%%'%(
        #         ': %s'%(ld_name if ld_name else ''), time.time()-t_start, error*100.0, ece*100.0))

        # return error, ece


