import os, sys
import numpy as np
from scipy import stats

import torch as tc

def compute_top_conf(mdl, ld, device):
    ph_list = []
    mdl = mdl.to(device)
    for x, y in ld:
        x, y = x.to(device), y.to(device)
        with tc.no_grad():
            out = mdl(x)
        ph = out['ph_top']
        ph_list.append(ph.cpu())
    ph_list = tc.cat(ph_list)
    return ph_list
    


## https://gist.github.com/DavidWalz/8538435
def bino_ci(k, n, alpha=1e-5):
    lo = stats.beta.ppf(alpha/2, k, n-k+1)
    hi = stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi
