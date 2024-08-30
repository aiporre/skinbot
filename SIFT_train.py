import numpy as np
import cv2 as cv

img = cv.imread('data/images/BLAND_1_IMAGIC_1905060485411.JPG')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray, None)

img = cv.drawKeypoints(gray, kp, img)

cv.imwrite('sift_keypoints.jpg', img)