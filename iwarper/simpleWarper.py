import numpy as np
import time

mxValue = 1000000.0

def estimatePixelCostVer(image):
    img = image.astype(np.float64) / 256.0
    h, _, _ = img.shape

    diff = np.sum(np.abs(img[:,0:-1,:] - img[:, 1:, :]), 2)

    cost = np.hstack([np.ones((h, 1), dtype=np.float64) * mxValue,
                      diff[:,:-1] + diff[:,1:], 
                      np.ones((h, 1), dtype=np.float64) * mxValue])
    return cost

def estimatePixelCostHor(image):
    img = image.astype(np.float64) / 256.0
    _, w, _ = img.shape

    diff = np.sum(np.abs(img[0:-1,:,:] - img[1:, :, :]), 2)
    cost = np.vstack([np.ones((1, w), dtype=np.float64) * mxValue,
                      diff[:-1,:] + diff[1:,:], 
                      np.ones((1, w), dtype=np.float64) * mxValue])
    return cost

def calculatePathsVer(cost):
    h, w = cost.shape

    x = np.zeros((h, w), dtype=np.int32)

    for i in range(1, h):
        dirs = np.vstack([cost[i-1, :-2], cost[i-1,1:-1], cost[i-1, 2:]])
        x[i, 1:-1] = np.argmin(dirs, axis = 0)
        cost[i, 1:-1] = cost[i, 1:-1] + dirs[x[i, 1:-1], np.arange(dirs.shape[1])]
        x[i, 1:-1] = x[i, 1:-1] - 1

    return x

def calculatePathsHor(cost):
    h, w = cost.shape

    x = np.zeros((h, w), dtype=np.int32)

    for i in range(1, w):
        dirs = np.hstack([cost[:-2, i-1].reshape(h-2, 1),
                          cost[1:-1,i-1].reshape(h-2, 1),
                          cost[2:,i-1].reshape(h-2, 1)])
        x[1:-1,i] = np.argmin(dirs, axis = 1)
        cost[1:-1, i] = cost[1:-1, i] + dirs[np.arange(dirs.shape[0]), x[1:-1, i]]
        x[1:-1, i] = x[1:-1, i] - 1

    return x

def deleteLineVer(image, cost, paths):
    h, _, _ = image.shape

    ind = np.argmin(cost[-1, :])

    new_image = image[:, :-1, :]
    for y in range(h-1,0,-1):
        new_image[y, ind:, :] = image[y, ind+1:,:]

        ind = ind + paths[y, ind]

    return new_image

def deleteLineHor(image, cost, paths):
    _, w, _ = image.shape

    ind = np.argmin(cost[:, -1])

    new_image = image[:-1, :, :]
    for x in range(w-1,0,-1):
        new_image[ind:, x, :] = image[ind+1:, x, :]

        ind = ind + paths[ind, x]

    return new_image

def warpVertical(image):
    cost = estimatePixelCostVer(image)
    paths = calculatePathsVer(cost)
    return deleteLineVer(image, cost, paths)

def warpHorizontal(image):
    cost = estimatePixelCostHor(image)
    paths = calculatePathsHor(cost)
    return deleteLineHor(image, cost, paths)

def warp(image, rows = 0, cols = 0):

    image_copy = image.copy()

    rc = min(rows, cols)

    for i in range(rc):
        image_copy = warpVertical(image_copy)
        image_copy = warpHorizontal(image_copy)

    for c in range(cols - rc):
        image_copy = warpVertical(image_copy)

    for r in range(rows - rc):
        image_copy = warpHorizontal(image_copy)

    return image_copy

def warpYield(image, rows = 5, cols = 10):

    image_copy = image.copy()

    rc = min(rows, cols)

    for i in range(rc):
        image_copy = warpVertical(image_copy)
        yield image_copy
        image_copy = warpHorizontal(image_copy)
        yield image_copy
        
    for c in range(cols - rc):
        image_copy = warpVertical(image_copy)
        yield image_copy

    for r in range(rows - rc):
        image_copy = warpHorizontal(image_copy)
        yield image_copy
