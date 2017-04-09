# Pipeline for Project 4 - Car ND - Advanced Lane detection

# Advanced Lane Finding Project
# The goals / steps of this project are the following:
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Apply a distortion correction to raw images.
# Use color transforms, gradients, etc., to create a thresholded binary image.
# Apply a perspective transform to rectify binary image ("birds-eye view").
# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline


calibration_path = '/Users/Akshay/projects/carnd/CarND-Advanced-Lane-Lines/camera_cal/'
img = mpimg.imread(calibration_path + "calibration1.jpg")
plt.imshow(img)
