import numpy as np
from model import Blur

kernel = Blur._gkern2(kernlen=3, nsig=1/1.25)
print(kernel)

desired = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
desired = desired/np.sum(desired).astype('float32')
print(desired)
