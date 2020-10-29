# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:40:15 2020

@author: 91948
"""
import cv2
import numpy as np
import glob
import os

cap = cv2.VideoCapture(0)
ctr = 0
max_ctr = 200
while(True):
    # Capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
	if ret == True:
		print (ctr)
		cv2.imwrite("picture" + str(ctr) + ".jpg", frame)
		ctr = ctr + 1
		frame = cv2.drawChessboardCorners(frame, (8,6), corners, ret)
	if ctr > max_ctr:
		break
    # Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')
ctr = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(1)
   

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print (mtx, dist)
np.savez("Camera.npz", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
for frame in images:
    os.remove(frame)

cv2.destroyAllWindows()
