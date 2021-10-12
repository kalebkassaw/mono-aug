import numpy as np
import cv2
# import albumentations as A
from functools import partial
from skimage.exposure import match_histograms

def load_image(fname, gray=True):
    if gray:
        return cv2.imread(fname, 0)
    else:
        return cv2.imread(fname)

class MonoFunc:
    def __init__(self, mode='int', gray=True):
        self.float = mode != 'int'
        self.period = 1 if self.float else 255

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

    def hm(self, x, y):
        return match_histograms(x, y)

class Wiggle:
    def __init__(self, mode='int', gray=True):
        self.float = mode != 'int'
        self.period = 1 if self.float else 255
        self.gray = gray

    def __call__(self, x, dev, save_map=None):
        if not isinstance(x, np.ndarray):
            x = load_image(x, self.gray)
        if self.gray:
            return self.wiggle_1ch(self, x, dev, save_map)
        else:
            return self.wiggle_3ch(self, x, dev, save_map)

    def wiggle_map(self, dev, save_map=None):
        map_out = np.zeros(256)
        rand = np.random.randint(-dev, dev + 1)
        for i in range(len(map_out)):
            if i == 0: 
                continue
            if i == 255:
                map_out[i] = 255
                break
            if rand > 0:
                map_out[i] = 1 + rand + map_out[i-1]
                rand = np.random.randint(-dev, dev + 1)
            elif rand < 0:
                rand += 1
                map_out[i] = map_out[i-1]
            else:
                rand = np.random.randint(-dev, dev + 1)
                map_out[i] = 1 + map_out[i-1]
            # print(i, "map_out[i]" , map_out[i])
        map_out[map_out > 255] = 255
        if save_map is not None:
            np.save(save_map, map_out)
        return map_out

    def wiggle_1ch(self, x, dev, save_map=None):
        x = x[:, :, 0]
        wmap = self.wiggle_map(dev, save_map)
        out = np.ravel(x)
        out = [wmap[i] for i in out]
        out = np.array(out).astype(int).reshape(x.shape)
        return out

    def wiggle_3ch(self, x, dev, save_map=None):
        wmap = self.wiggle_map(dev, save_map)
        out = [np.ravel(x[:,:,i]) for i in range(3)]
        out = [[wmap[i] for i in a] for a in out]
        out = [a.reshape(x[:,:,0].shape) for a in out]
        out = np.stack(out, axis=-1).astype(int)
        return out

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