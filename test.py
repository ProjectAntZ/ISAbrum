import random

import numpy as np

x = np.arange(100)
x = np.reshape(x, (10, 10))
x = x == 50
x = x.astype('uint8')
x *= 255
print(x)

