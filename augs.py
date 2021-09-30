import numpy as np
import cv2
import albumentations as A
from functools import partial

class MonoAug:
    def __init__(self, mode='int'):
        self.float = mode != 'int'
        self.period = 1 if self.float else 255

    def load_image(self, fname, mode='gray'):
        if mode == 'gray':
            return cv2.imread(fname, 0)
        else:
            return cv2.imread(fname)
    
    def norm(self, x):
        if len(x.shape) == 3:
            max = np.amax(x, axis=2)
            min = np.amin(x, axis=2)
        else:
            max = np.amax(x)
            min = np.amin(x)
        oa = np.ones_like(x)
        out = (x - min * oa) / (max - min)
        out = out * self.period
        return out

    '''
    static method sin/tan:
    arguments: 
        x, an image
        a, a constant multiple of a period
    '''
    def sin(self, x, a):
        if a < 1:
            np.sin(a * np.pi * x/(self.period * 2))
        else:
            return np.sin(np.pi * x/(self.period * 2))

    def tan(self, x, a):
        if a < 1:
            return np.tan(a * np.pi * x/(self.period * 4))
        else:
            return np.tan(np.pi * x/(self.period * 4))

    '''
    static method poly:
    arguments: 
        x, an image
        a, a degree
    '''
    def poly(self, x, a): 
        return np.power(x, a * np.ones_like(x))

    def random(self, x, num_ops):
        choices = ['poly', 'sinu']
        sinuwaves = ['sin', 'tan']
        choice = choices[np.random.randint(0, 2)]
        for _ in range(num_ops):
            if choice == 'poly':
                exp = np.random.randint(0, 6)
                x = self.poly(x, exp)
            if choice == 'sinu':
                wave = sinuwaves[np.random.randint(0, 2)]
                pd = np.random.uniform(0.5, 1.0)
                if wave == 'sin':
                    x = self.sin(x, pd)
                if wave == 'tan':
                    x = self.tan(x, pd)
        x = self.norm(x)
