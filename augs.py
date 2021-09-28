
import numpy as np

class MonoAug:
    def __init__(mode='int'):
        self.float = mode != 'int'
    
    ## image augmentations that are monotonic functions

    ## polynomials x^a, a integer greater than 0, odd
    poly: lambda x, a: np.power(x, a * np.ones_like(x))

    ## sinusoids with period greater than 1024:
    sin: lambda x, a: np.sin(a * np.pi * x/512) if a < 1 else (np.sin(pi * x/512))
    tan: lambda x, a: np.sin(a * np.pi * x/1024) if a < 1 else (np.sin(pi * x/1024))