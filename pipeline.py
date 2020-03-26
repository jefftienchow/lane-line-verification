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
from conv import convolve
from shape import shape
from evaluator import get_size


# np.set_printoptions(threshold=sys.maxsize)
SCALE_FACTOR = 1/5
X_CROP = 150
SOURCE_PTS = [(580, 460), (205, 720), (1110, 720), (703, 460)]
DEST_PTS = [(320, 0), (320, 720), (960, 720), (960, 0)]
PAINTED_THRESHOLDS = {'sat_thresh': 160, 'light_thresh': 120, 'light_thresh_agr': 240,
             'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20}
REG_THRESHOLDS = { 'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
              'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20 }

def offset(lines, rand = True, factor = 1):
    """
    Moves the given lines from their original position
    :param lines: the given lines as 2nd degree polynomial coefficients
    :param rand: whether to use a random offset
    :param factor: the randomization factor
    :return: the moved lines
    """
    for line in lines:
        if rand:
            line[2] = line[2] + factor * (random.random() - .5) * 100
            line[1] = line[1] + factor * (random.random() - .5) * .2
            line[0] = line[0] + factor * (random.random() - .5) * .0001
        else:
            line[2] = line[2] + factor * 100
            line[1] = line[1] + factor * -.3
            line[0] = line[0] + factor * .0001

def get_wb(img, birds_eye, thresholds):
    """
    Applies transformations to process the given image
    :param img: the given image
    :param birds_eye: the bird's eye transformation object
    :param thresholds: thresholds used for vision filters
    :return: the processed image
    """
    lane_filter = LaneFilter(thresholds)
    binary = lane_filter.apply(img)
    plt.imshow(binary)
    plt.show()

    # birds_eye = cv2.warpPerspective(binary, warp, (img.shape[1], img.shape[0]), flags = cv2.INTER_LINEAR)

    birds_eye_view = birds_eye.sky_view(binary)
    plt.imshow(birds_eye_view)
    # plt.show()
    width = birds_eye_view.shape[1]
    return roi(birds_eye_view, width//10, width - width//10).astype(np.uint8)

def get_lines(img, birds_eye, thresholds):
    """
    Finds the lane lines in the given image
    :param img: the image to find the lane lines on
    :param birds_eye: the bird's eye transformation object
    :param thresholds: thresholds used for vision filters
    :return: the coefficients of the detected left and right lane lines as 2nd degree polynomials
    """
    wb = get_wb(img, birds_eye, thresholds)
    curves = Curves(number_of_windows=9, margin=100, minimum_pixels=10,
                    ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)
    result = curves.fit(wb)
    plt.imshow(result['image'])
    plt.show()
    visual = birds_eye.project(img, wb, result['pixel_left_best_fit_curve'], result['pixel_right_best_fit_curve'])
    plt.imshow(visual)
    plt.show()

    return (result['pixel_left_best_fit_curve'], result['pixel_right_best_fit_curve'])

def resize(img, lines, source_pts, dest_pts, scale_factor):
    """
    resizes parameters to reduce the size of the certificate
    """
    print(img.shape)
    # half_height = img.shape[0]//2
    # width = img.shape[1]
    # img = img[half_height:, x_crop:width-x_crop]
    img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))
    plt.imshow(img)
    plt.show()
    for line in lines:
        line[0] /= scale_factor
        line[2] = (line[2]) * scale_factor
    source_pts = [(pt[0] * scale_factor, pt[1] * scale_factor) for pt in source_pts]
    dest_pts = [(pt[0] * scale_factor, pt[1] * scale_factor) for pt in dest_pts]
    print(source_pts)
    print(img.shape)
    plt.imshow(img)
    plt.show()
    return img, lines, source_pts, dest_pts

def interlock(img, birds_eye, thresholds, lines):
    '''
    runs the actual vision interlock; the parameters are the components that create the `certificate`
    :param img: image from the controller
    :param birds_eye: bird's eye transformation object from the controller
    :param thresholds: thresholds used for vision filters
    :param lines: proposed lane lines
    :return: the result of each vision interlock test
    '''

    wb = get_wb(img, birds_eye, thresholds)
    plt.imshow(wb)
    plt.show()
    shape_result = shape(lines[0], lines[1], img.shape[0])
    left_result = convolve(True, lines[0], wb)
    right_result = convolve(False, lines[1], wb)

    result = birds_eye.project(img, wb, lines[0], lines[1])

    plt.title("Approach 1: " + ("pass" if shape_result else "fail") + "\nApproach 2L: " +
              (left_result if left_result else "fail") + ", Approach 2R: " + (right_result if right_result else "fail"))
    plt.imshow(result)
    plt.show()
    return shape_result, left_result, right_result

def crop(img, source_pts, dest_pts, top_crop, x_crop):
    source_pts = [(pt[0] - x_crop, pt[1] - top_crop) for pt in source_pts]
    dest_pts = [(pt[0] - x_crop, pt[1] - top_crop) for pt in dest_pts]
    img = img[top_crop:, x_crop:img.shape[1] - x_crop]
    print("after crop shape is: ", img.shape)
    print(source_pts)
    plt.imshow(img)
    plt.show()
    return img, source_pts, dest_pts

def pipeline(img, lines=None, move_lines=False):
    '''
    pipeline for simulating the controller and interlock
    :param img: The image to be processed
    :param lines: proposed lane lines; if not given, a default lane-finding algorithm will generate proposed lines
    :param move_lines: whether or not to alter the proposed lane lines
    :return: whether or not the final proposed lane lines pass
    '''
    # RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, source_pts, dest_pts = crop(img, SOURCE_PTS, DEST_PTS, img.shape[0]//2, 150)
    birds_eye = BirdsEye(source_pts, dest_pts)
    if not lines:
        lines = get_lines(img, birds_eye, REG_THRESHOLDS)
    if move_lines:
        offset(lines, True, 1)

    img, lines, source_pts, dest_pts = resize(img, lines, source_pts, dest_pts, SCALE_FACTOR)

    interlock_birdseye = BirdsEye(source_pts, dest_pts)
    print("Size is: ", get_size(img))
    print(get_size(interlock_birdseye))
    print(get_size(REG_THRESHOLDS))
    print(get_size(lines))
    shape_result, left_result, right_result = interlock(img, interlock_birdseye, REG_THRESHOLDS, lines)
    return shape_result and left_result and right_result

if __name__ == "__main__":
    print(pipeline(cv2.imread('test_images/test1.jpg')))
