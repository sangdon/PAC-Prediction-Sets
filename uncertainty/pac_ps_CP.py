import os, sys
import numpy as np
import pickle
import types
import itertools
import scipy
import math
import warnings

import torch as tc

from learning import *
from uncertainty import *
import model
from .util import *
    

class PredSetConstructor_CP(PredSetConstructor):
    def __init__(self, model, params=None, name_postfix=None):
        super().__init__(model=model, params=params, name_postfix=name_postfix)

        
    def train(self, ld, params=None):
        if params is None:
            params = self.params
        m = params.n
        eps = params.eps
        delta = params.delta
        if self.name_postfix is None:
            self.name_postfix = ''    
        self.name_postfix = self.name_postfix + f'n_{m}_eps_{eps:e}_delta_{delta:e}'
        verbose = params.verbose
        #m, eps, delta = self.mdl.n.item(), self.mdl.eps.item(), self.mdl.delta.item()
        print(f"## construct a prediction set: m = {m}, eps = {eps:.2e}, delta = {delta:.2e}")

        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True

        ## precompute -log f(y|x)
        f_nll_list = []
        for x, y in ld:
            x, y = to_device(x, self.params.device), to_device(y, self.params.device)

            f_nll_i = self.mdl(x, y)
            f_nll_list.append(f_nll_i)
            if m <= sum([len(v) for v in f_nll_list]):
                break
        f_nll_list = tc.cat(f_nll_list)
        f_nll_list = f_nll_list[:m]
        #assert len(f_nll_list) == m, f'len(f_nll_list) = {len(f_nll_list)} != m = {m}'

        ## line search over T
        T, T_step, T_end, T_opt_nll = 0.0, self.params.T_step, self.params.T_end, np.inf
        while T <= T_end:
            T_nll = -np.log(T).astype(np.float32) if T>0 else np.inf
            ## CP bound
            error_U = (f_nll_list > T_nll).sum().float()
            k_U, n_U, delta_U = error_U.int().item(), m, delta
            #print(f'[Clopper-Pearson parametes] k={k_U}, n={n_U}, delta={delta_U}')
            U = bci_clopper_pearson_worst(k_U, n_U, delta_U)

            if U <= eps:
                T_opt_nll = T_nll
            elif U >= self.params.eps_tol*eps: ## no more search if the upper bound is too large
                break
            if verbose:
                print(f'[m = {m}, eps = {eps:.2e}, delta = {delta:.2e}, T = {T:.4f}] '
                      f'T_opt = {math.exp(-T_opt_nll):.4f}, #error = {k_U}, error_emp = {k_U/n_U:.4f}, U = {U:.6f}')

            T += T_step        

        print(f'T_opt = {math.exp(-T_opt_nll):.8f}')
        ## save parameters
        self.mdl.T.data = tc.tensor(T_opt_nll)
        self.mdl.n.data = tc.tensor(m)
        self.mdl.eps.data = tc.tensor(eps)
        self.mdl.delta.data = tc.tensor(delta)
        self.mdl.to(self.params.device)

        ## save models
        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
    
