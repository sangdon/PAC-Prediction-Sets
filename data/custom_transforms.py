import os, sys
import torch as tc


class Identity:
    def __call__(self, x):
        return x

class ToJustTensor:
    def __call__(self, x):
        return tc.tensor(x)

    
class Normalizer:
    def __init__(self, n):
        self.n = tc.tensor(n)

        
    def __call__(self, x):
        return x / self.n
        
