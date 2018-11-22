import numpy as np
import time

import fastWarper

def warp(image, rows = 0, cols = 0):

    image_copy = image.astype(np.float64) / 256.0
    image_copy = fastWarper.warp(image_copy, cols, rows) * 256

    if rows > 0 and cols > 0:
        image_copy = image_copy[:-rows, :-cols, :]
    elif rows > 0:
        image_copy = image_copy[:-rows, :, :]
    elif cols > 0:
        image_copy = image_copy[:, :-cols :]

    return image_copy.astype(np.uint8)