import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle

from birdseye import BirdsEye
from lanefilter import LaneFilter
from curves import Curves
from helpers import roi
import sys
from scipy import signal
import random


def resize(img, scale_factor):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim)

def convolve(left, fit, img):
    if left:
        print('starting convolution on left')
    else:
        print('starting convolution on right')
    scale_factor = 1/4

    print('resizing image...')
    img = resize(img, scale_factor)

    # plt.imshow(img)
    # plt.show()

    print('creating filter...')
    filter_fit = np.copy(fit)
    filter_fit[0] *= scale_factor
    filter_fit[2] /= scale_factor
    h = img.shape[0]
    xp = np.arange(0, h - 1)

    filter_values = np.arange(100, 255, 155/(h-1))

    filter_p = np.poly1d(filter_fit)

    yp = filter_p(xp)
    min_y = np.amin(yp)
    w = int(np.amax(yp) - np.amin(yp))

    filter = np.full((h + 10, w + 10), 0).astype(np.uint8)

    for i in range(len(filter_values)):
        filter[(xp[i] + 5).astype(int), (yp[i] - min_y + 5).astype(int)] = filter_values[i]

    blur = np.full([4, 4], 1 / 16)
    filter = signal.convolve2d(filter, blur)

    # plt.imshow(filter)
    # plt.show()

    print('boundary is at: ',int(img.shape[1]/2))

    half_width = int(img.shape[1]/2)
    if left:
        img = img[:, :half_width]
    else:
        img = img[:, half_width + 1:]
    #
    # plt.imshow(img)
    # plt.show()


    grad = signal.convolve2d(img, filter, 'same')
    # plt.imshow(grad, cmap = 'gray')
    # plt.show()


    result = np.where(grad == np.amax(grad))
    # print('result is: ', result)
    result_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(5):
        for j in range(5):
            result_img[i+result[0][0]][j+result[1][0]] = 255
    print('max val is: ', grad[result[0][0]][result[1][0]])
    p = np.poly1d(fit)

    if left:
        actual_x = result[1][0] * (1/scale_factor)
        expected_x = p(result[0][0] * (1/scale_factor))
    else:
        actual_x = (result[1][0] + half_width)* (1/scale_factor)
        expected_x = p(result[0][0] * (1/scale_factor))


    print('expected: ', expected_x)
    print('actual: ', actual_x)

    # plt.imshow(result_img)
    # plt.show()

    if abs(actual_x - expected_x) < 25 and grad[result[0][0]][result[1][0]] > 7500:
        return True
    else:
        return False