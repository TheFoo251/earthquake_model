from PIL import Image
import numpy as np
from sys import argv as args

for path in args[1:]:
    with Image.open(path) as im:
        im_array = np.asarray(im)
    damaged_pixels = np.argwhere(im_array > 2)
    if damaged_pixels.size:
        print(path)
