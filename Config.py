import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.SCALES = (600,)
__C.MAX_SIZE = 1000
