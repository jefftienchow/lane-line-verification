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
import math


scale_factor = 1 / 4
# np.set_printoptions(threshold=sys.maxsize)



def get_dotted_filter(fit, funct):
    dot_len = 30 #* scale_factor
    space_len = 60 #* scale_factor
    # print('fit is', fit)
    degree = math.atan(fit[1])
    print('degree is: ', degree)
    dot_range = int(math.cos(degree) * dot_len)
    space_range = int(math.cos(degree) * space_len)
    print("dot range is: ", dot_range)
    print("dot range is: ", space_range)


    dot_y = np.concatenate([np.arange(0, dot_range), np.arange(dot_range + space_range, dot_range*2 + space_range)])
    space_y = np.arange(dot_range, dot_range + space_range)

    dotx = funct(dot_y)
    spacex = funct(space_y)

    h = dot_range*2 + space_range
    min_x = min(np.amin(dotx), np.amin(spacex))
    w = int(max(np.amax(dotx), np.amax(spacex)) - min_x)
    print('dimensions are: ', w , " ", h)

    filter = np.full((h + 10, w + 10), 0).astype(np.int16)

    cur_val = 100
    loss = 155/len(dot_y)
    for i in range(len(dot_y)):
        filter[(dot_y[i] + 5).astype(int), (dotx[i] - min_x + 5).astype(int)] = cur_val
        cur_val += loss

    # plt.imshow(filter)
    # plt.show()
    cur_val = -100

    loss = 50/len(space_y)
    for i in range(len(space_y)):
        filter[(space_y[i] + 5).astype(int), (spacex[i] - min_x + 5).astype(int)] = cur_val
        cur_val -= loss
    # print(filter)
    blur = np.full([4, 4], 1 / 16)
    filter = signal.convolve2d(filter, blur)
    plt.imshow(filter)
    # plt.show()
    return filter

# def resize(img, scale_factor):
#     width = int(img.shape[1] * scale_factor)
#     height = int(img.shape[0] * scale_factor)
#     dim = (width, height)
#     # resize image
#     return cv2.resize(img, dim)

def convolve(left, fit, img):
    if left:
        print('starting convolution on left')
    else:
        print('starting convolution on right')

    # print('resizing image...')
    # img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    # plt.imshow(img)
    # plt.show()

    print('creating filter...')

    # copies the fit and scales it
    filter_fit = np.copy(fit)
    # filter_fit[0] /= scale_factor
    # filter_fit[2] *= scale_factor
    h = img.shape[0]
    ypts = np.arange(0, h - 1)

    # filter values are heavier towards the bottom of the filter
    filter_values = np.arange(100, 255, 155/(h-1))

    filter_p = np.poly1d(filter_fit)

    xpts = filter_p(ypts)

    min_x = np.amin(xpts)
    w = int(np.amax(xpts) - np.amin(xpts))

    padding = 10
    filter = np.full((h + padding, w + padding), 0).astype(np.uint8)

    for i in range(len(filter_values)):
        # subtracts min_x to account for offset given by fit
        filter[(ypts[i] + padding/2).astype(int), (xpts[i] - min_x + padding/2).astype(int)] = filter_values[i]

    blur = np.full([4, 4], 1 / 16)
    filter = signal.convolve2d(filter, blur)

    plt.imshow(filter)
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

    dotted = get_dotted_filter(filter_fit, filter_p)
    grad = signal.correlate2d(img, filter, 'same')

    grad2 = signal.correlate2d(img, dotted, 'same')

    plt.imshow(grad, cmap='gray')
    # plt.show()
    plt.imshow(grad2, cmap = 'gray')
    # plt.show()

    dotted_result = np.unravel_index(grad2.argmax(), grad2.shape)
    # dotted_result = np.where(grad2 == np.amax(grad2))
    print(grad2)
    print('max val for dotted is: ', grad2[dotted_result[0]][dotted_result[1]])

    result = np.unravel_index(grad.argmax(), grad.shape)
    # print('result is: ', result)
    # result_img = np.zeros((img.shape[0], img.shape[1]))
    # for i in range(5):
    #     for j in range(5):
    #         result_img[i+result[0][0]][j+result[1][0]] = 255
    print('max val is: ', grad[result[0]][result[1]])
    print('location of max for solid is: ', result)
    print('location of max for dotted is: ', dotted_result)

    p = np.poly1d(fit)

    offset = int(filter.shape[1] / 2) - (xpts[result[0]] - min_x + padding / 2)

    if left:
        print('actual middle x is: ', result[1])
        print("filter shape is: ", filter.shape)
        print('xpts[result[0]] is: ', xpts[result[0]])

        actual_x = (result[1] - offset) #* (1/scale_factor)
        expected_x = p(result[0] )#* (1/scale_factor))
        actual_2x = dotted_result[1] #* (1/scale_factor)
        expected_2x = p(dotted_result[0])# * (1/scale_factor))

    else:
        actual_x = (result[1] + half_width - offset)#* (1/scale_factor)
        expected_x = p(result[0] )#* (1/scale_factor))

        actual_2x = (dotted_result[1] + half_width) #* (1/scale_factor)
        expected_2x = p(dotted_result[0])# * (1/scale_factor))




    print('expected: ', expected_x)
    print('expected_2:', expected_2x)
    print('expected scaled: ', filter_p(result[0]))
    print('actual: ', actual_x)
    print('actual2: ', actual_2x)

    # plt.imshow(img)
    # plt.show()

    if abs(actual_x - expected_x) < 20 and grad[result[0]][result[1]] > 13000:
        print('its solid')
        return "solid"
    elif abs(actual_2x - expected_2x) < 20 and grad2[dotted_result[0]][dotted_result[1]] > 900:
        print('its dotted')
        return "dotted"
    return False