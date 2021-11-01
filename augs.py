import numpy as np
import cv2
# import albumentations as A
from functools import partial
from skimage.exposure import match_histograms
import scipy.stats as ss

def load_image(fname, gray=True):
    if gray:
        a = cv2.imread(fname, 0)
        return np.stack((a,a,a), axis=-1)
    else:
        return cv2.imread(fname)

#TODO: fix this function to work more accurately -- flattening behavior should not be happening
def _match_cumulative_cdf(image, mapgen):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    if len(image.shape) == 3: im = np.ravel(image[:,:,0])
    else: im = np.ravel(image)
    im_vals, im_indices, im_counts = np.unique(im, return_inverse=True, return_counts=True) # calculate normalized quantiles for each array
    im_quantiles = np.cumsum(im_counts) / im.size
    map_quantiles = np.cumsum(mapgen) / mapgen.size
    interp_a_values = np.interp(im_quantiles, map_quantiles, mapgen)
    im = interp_a_values[im_indices].reshape(image.shape[:2])
    if len(image.shape) == 3: im = np.stack((im,im,im), axis=-1)
    im = im.astype(int)
    return im

class InfoLoss:
    def __init__(self, gamma=2.0):
        self.gamma = gamma

    def __call__(self, orig, mod):
        loss, fail = self.info_loss(orig, mod)
        return fail

    def get_histogram(self, image):
        return np.bincount(image.reshape(-1), minlength=256)

    def get_entropy(self, image):
        hist = self.get_histogram(image)
        dist = hist / np.sum(hist)
        return ss.entropy(dist, base=2)

    def info_loss(self, I, I_p):
        d_entropy = 0
        for c in range(3):
            d_entropy += self.get_entropy(I[:, :, c]) - self.get_entropy(I_p[:, :, c])
        loss = d_entropy / 3
        fail = loss > self.gamma
        return loss, fail

class AtNorm:
    def __init__(self, ll1=0.1, ul1=99.9, ll2=-5, ul2=10):
        self.lower_lim1 = ll1
        self.upper_lim1 = ul1
        self.lower_lim2 = ll2
        self.upper_lim2 = ul2

    def __call__(self, x):
        #print(x.shape)
        x = self._clip1(x)
        #print(x.shape)
        x = self._normalize_mad(x)
        #print(x.shape)
        x = self._clip2(x)
        #print(x.shape)
        return x

    def _normalize_mad(self, x):
        med = np.median(x[:,:,2], axis=None) if len(x.shape) == 3 else np.median(x, axis=None)
        if len(x.shape) == 3:
            mad = ss.median_abs_deviation(x[:,:,2], axis=None)
        else:
            mad = ss.median_abs_deviation(x, axis=None)
        if mad < 1.0e-5: mad = 1.0e-5
        x = (x - med) / mad
        return x

    def _clip1(self, x, lower_lim=0.1, upper_lim=99.9):
        dn, up = np.percentile(x, [self.lower_lim1, self.upper_lim1])
        x = np.clip(x, dn, up)
        x -= np.amin(x)
        x *= 255 / np.amax(x)
        return x

    def _clip2(self, x, lower_lim=-5, upper_lim=10):
        x = np.clip(x, self.lower_lim2, self.upper_lim2)
        x -= np.amin(x)
        x *= 255 / np.amax(x)
        #print(x)
        return x


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
        else:
            a = x[:,:,0]
            x = np.stack((a,a,a), axis=-1)
        # return self.remap(x, dev, save_map)
        return self.wiggle_3ch(x, dev)


    def rand_generator(self, dev, mode):
        if mode == 'unif':
            return np.random.randint(-dev, dev + 1)
        elif mode == 'norm':
            return np.around(np.random.normal(1, dev), decimals=0).astype(int)
        else:
            return np.around(np.random.laplace(1, dev), decimals=0).astype(int)

    def wiggle_map(self, dev, mode='lapl', save_map=None):
        map_out = np.zeros(256)
        rand = self.rand_generator(dev, mode)
        for i in range(len(map_out)):
            if i == 0: 
                continue
            if i == 255:
                map_out[i] = 255
                break
            if rand > 0:
                map_out[i] = rand + map_out[i-1]
                rand = self.rand_generator(dev, mode)            
            elif rand < 0:
                rand += 1
                map_out[i] = map_out[i-1]
            else:
                rand = self.rand_generator(dev, mode)
                map_out[i] = map_out[i-1]
            # print(i, "map_out[i]" , map_out[i])
        map_out[map_out > 255] = 255
        if save_map is not None:
            np.save(save_map, map_out)
        return map_out

    def remap(self, x, dev, save_map=None):
        wmap = self.wiggle_map(dev, save_map)
        out = _match_cumulative_cdf(x, wmap)
        return out

    '''def wiggle_1ch(self, x, dev, save_map=None):
        if len(x.shape()) == 3: x = x[:,:,0]
        wmap = self.wiggle_map(dev, save_map)
        out = np.ravel(x)
        out = [wmap[i] for i in out]
        out = np.array(out).astype(int).reshape(x.shape)
        return out'''

    def wiggle_3ch(self, x, dev, save_map=None):
        if self.gray: 
            x[:,:,1] = x[:,:,0]
            x[:,:,2] = x[:,:,0]
        wmap = self.wiggle_map(dev, save_map)
        out = [np.ravel(x[:,:,i]) for i in range(3)]
        out = [np.array([wmap[i] for i in a]) for a in out]
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
