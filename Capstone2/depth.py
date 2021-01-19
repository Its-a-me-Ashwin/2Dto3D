from arucoDetect import detect_markers,getCameraMatrix
import PIL
import cv2
import numpy as np
from matplotlib import pyplot as plt


cam, dist = getCameraMatrix()

img1 = plt.imread('1.jpg')
img2 = plt.imread('2.jpg')

data1 = detect_markers(img1,cam,dist)
data2 = detect_markers(img2,cam,dist)

print(data2)


rvec2 = data2[0][2][0][0]
rvec2Degree = list(map(lambda x: x*(180/3.141),rvec2))
print(rvec2Degree)