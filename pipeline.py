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

#np.set_printoptions(threshold=sys.maxsize)


from moviepy.editor import VideoFileClip
from IPython.display import HTML

calibration_data = pickle.load(open("calibration_data.p", "rb" ))

matrix = calibration_data['camera_matrix']
distortion_coef = calibration_data['distortion_coefficient']

source_points = [(580, 460), (205, 720), (1110, 720), (703, 460)]
destination_points = [(320, 0), (320, 720), (960, 720), (960, 0)]

p = { 'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
      'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20 }

birdsEye = BirdsEye(source_points, destination_points, matrix, distortion_coef)
laneFilter = LaneFilter(p)
curves = Curves(number_of_windows = 9, margin = 100, minimum_pixels = 50,
                ym_per_pix = 30 / 720 , xm_per_pix = 3.7 / 700)

def randomize(curve, rand = True, factor = 1):
    if rand:
        curve[2] = curve[2] + factor * (random.random() - .5) * 100
        curve[1] = curve[1] + factor * (random.random() - .5) * .2
        curve[0] = curve[0] + factor * (random.random() - .5) * .0001
    else:
        curve[2] = curve[2] + factor * 100
        curve[1] = curve[1] + factor * -.3
        curve[0] = curve[0] + factor * .0001


def test_wrong_fit(img):
    noise_factor = 1

    ground_img = birdsEye.undistort(img)
    binary = laneFilter.apply(ground_img)

    # plt.imshow(birdsEye.sky_view(ground_img))
    # plt.show()
    # print(binary)
    # plt.imshow(binary)
    # plt.show()
    # print('binary has been shown')

    birds_eye = birdsEye.sky_view(binary)
    wb = roi(birds_eye).astype(np.uint8)

    result = curves.fit(wb)

    print(result['pixel_right_best_fit_curve'])
    randomize(result['pixel_left_best_fit_curve'], False, noise_factor)
    randomize(result['pixel_right_best_fit_curve'], False, noise_factor)


    ground_img_with_projection = birdsEye.project(ground_img, binary,
                                                  result['pixel_left_best_fit_curve'],
                                                  result['pixel_right_best_fit_curve'])

    shape_result = shape(result['pixel_left_best_fit_curve'], result['pixel_right_best_fit_curve'], img)
    # plt.imshow(ground_img_with_projection)
    # plt.show()

    left_result = convolve(True, result['pixel_left_best_fit_curve'], wb)
    right_result = convolve(False, result['pixel_right_best_fit_curve'], wb)
    return left_result and right_result and shape_result


print(test_wrong_fit(cv2.imread('test_images/test3.jpg')))
