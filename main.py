import cv2
import imageio
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import time

import iwarper.simpleWarper as sWarper
import iwarper.fastWarper as fWarper

def adjustSize(im, newSize):
    hi,wi,_ = im.shape

    resim = np.zeros(newSize, dtype=np.uint8)
    resim[:hi,:wi,:] = im

    return resim

def showSimple(image, res):
    return plt.imshow(res)

def showOriginal(image, res):
    return plt.imshow(np.vstack([image, adjustSize(res, image.shape)]))

def showWithResized(image, res):
    resized = cv2.resize(image, (res.shape[1], res.shape[0]))
    return plt.imshow(np.vstack([np.hstack([image, image]),
                                 np.hstack([adjustSize(resized, image.shape),
                                 adjustSize(res, image.shape)])]))

def animate(image, cols, rows):
    obj = plt.figure()
    im = plt.imshow(image)

    def run(d):
        print(d.shape)
        im.set_data(adjustSize(d, image.shape))

    ani = animation.FuncAnimation(obj, run,
                                  sWarper.warpYield(image, cols = cols, rows = rows),
                                  repeat=False)
    plt.show()

def render(image, cols, rows):
    res = sWarper.warp(image, cols = cols, rows = rows)
    showOriginal(image, res)
    plt.show()

def testCext():
    import fastWarper

    print(fastWarper.check())

    x = np.arange(0, 2 * np.pi, 0.1)
    y = fastWarper.func_np(x)
    print(y)

    w = np.zeros((3, 3))
    w[1, 1] = 4
    v = fastWarper.func_np(w)
    print(v)

def measureSimpleWarperPerformance(image):
    start = time.time()
    sWarper.warp(image, cols = 10, rows = 0)
    print('Column time is {0} seconds'.format(time.time() - start))

    start = time.time()
    sWarper.warp(image, cols = 0, rows = 10)
    print('Row time is {0} seconds'.format(time.time() - start))

def compare2(image, cols, rows):
    start = time.time()
    res2 = fWarper.warp(image, cols = cols, rows = rows)
    print('Fast time is {0} seconds'.format(time.time() - start))

    start = time.time()
    res1 = sWarper.warp(image, cols = cols, rows = rows)
    print('Simple time is {0} seconds'.format(time.time() - start))

    plt.imshow(np.vstack([np.hstack([image, adjustSize(res1, image.shape)]),
                          np.hstack([adjustSize(res2, image.shape), image])]));
    plt.show()

#filename = './tests/data/dude.jpg'
#filename = './tests/data/squirrels.jpg'
#filename = './tests/data/desert.jpg'
filename = './tests/data/giraffes.jpg'

image = imageio.imread(filename)
columns = 150
rows = 100
compare2(image, cols = columns, rows = rows)

#render(image, cols = columns, rows = rows)
#animate(image, cols = columns, rows = rows)

print('End of main.py')
