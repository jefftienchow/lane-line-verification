import numpy as np
import math
import matplotlib.pyplot as plt

def scale(leftpoints, rightpoints, topy, bottomy):
    minim = min(leftpoints.min(), rightpoints.min())
    maxim = max(leftpoints.max(), rightpoints.max())
    leftpoints = (leftpoints - minim)/(maxim-minim) * (topy-bottomy) + bottomy
    rightpoints = (rightpoints - minim) / (maxim - minim) * (topy - bottomy) + bottomy
    return leftpoints, rightpoints

def leftShiftDown(leftpoints, rightpoints):
    shift = 0
    for (x,y) in zip(leftpoints, rightpoints):
        diff = x-y
        shift += diff
    return leftpoints - (shift / min(len(leftpoints), len(rightpoints)))

def getRMSDiff(leftpoints, rightpoints):
    rmsdiff = 0
    for (x, y) in zip(leftpoints, rightpoints):
        rmsdiff += (x - y) ** 2
    return math.sqrt(rmsdiff / min(len(leftpoints), len(rightpoints)))

def shape(left_fit, right_fit, img):
    h = img.shape[0]

    kl, kr = left_fit, right_fit
    ys = np.linspace(0, h - 1, h)
    lxs = kl[0] * (ys ** 2) + kl[1] * ys + kl[2]
    rxs = kr[0] * (ys ** 2) + kr[1] * ys + kr[2]

    leftpoints = lxs
    rightpoints = rxs
    x_coords = [-y for y in ys]
    topy, bottomy = 1, -1
    leftpoints, rightpoints = scale(leftpoints, rightpoints, topy,
                                    bottomy)
    l = leftShiftDown(leftpoints, rightpoints)
    rmsdiff = getRMSDiff(l, rightpoints)

    #plt.plot(x_coords, leftpoints)
    #plt.plot(x_coords, rightpoints)
    #plt.show()

    if rmsdiff < 0.5:
        return True
    return False