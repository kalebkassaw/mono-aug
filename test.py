import numpy as np

ones = 1 * np.ones((3, 3))
twos = 2 * np.ones((3, 3))
thrs = 3 * np.ones((3, 3))\

new = np.stack((ones, twos, thrs), axis=-1)
print(new)