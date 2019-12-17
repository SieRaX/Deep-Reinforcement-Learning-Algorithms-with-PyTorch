import numpy as np
from PIL import Image

b = np.load('episode1.npy')
print(b.shape)

c = b[0]
print(c.shape)

img = Image.fromarray(c, 'RGB')
img.show()