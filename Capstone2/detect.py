import numpy as np
import cv2
import cv2.aruco as aruco
import time,math,sys
from math import sqrt

expectedId = 6
marker_size = 13.5 # real measurements (meters)


# rotate matrix
R_flip = [
            [1.0,0.0,0.0],
            [0.0,-1.0,0.0],
            [0.0,0.0,-1.0]
        ]
R_flip = np.array(R_flip)

aruco_dict=aruco.Dictionary_get(aruco.DICT_5X5_250)
parameters=aruco.DetectorParameters_create()

def getCameraMatrix():
	with np.load('System.npz') as X:
		camera_matrix, dist_coeff, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
	return camera_matrix, dist_coeff

def isRotationMatrix(R):
    Rt = np.transpose(R)
    identity = np.dot(Rt,R)
    I = np.identity(3,dtype=R.dtype)
    n = np.linalg.norm(I-identity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert isRotationMatrix(R)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1],R[2,2])
        y = math.atan2(R[2,0],sy)
        z = math.atan2(R[1,0],R[0,0])
    else:
        x = math.atan2(-R[1,2],R[1,1])
        y = math.atan2(-R[2,0],sy)
        z = 0.0
    return np.array([x,y,z])
cam,dist = getCameraMatrix()


def getDistanceToMarker(x,y,z):
    if x == None or y == None or z == None:
        return None
    else:
        return  sqrt(x*x + y*y + z*z)

def getCameraPosition(image,expectedId,frame=None,marker_size = marker_size):
    cam,dist = getCameraMatrix()
    aruco_dict=aruco.Dictionary_get(aruco.DICT_5X5_250)
    parameters=aruco.DetectorParameters_create()
    corners,ids,_ = aruco.detectMarkers(image=image,dictionary = aruco_dict,parameters=parameters,
                                                cameraMatrix=cam,distCoeff=dist)
    if ids != None and expectedId in ids[0]:
        ret = aruco.estimatePoseSingleMarkers(corners,marker_size,cam,dist)

        rvec,tvec = ret[0][0,0,:],ret[1][0,0,:]     
        if True:
            aruco.drawDetectedMarkers(frame,corners)
            aruco.drawAxis(frame,cam,dist,rvec,tvec,marker_size//2)

        R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
        R_tc = R_ct.T

        #roll,pitch,yaw = rotationMatrixToEulerAngles(R_flip*R_tc)
        #print("X=%4.3f Y=%4.3f Z=%4.3f Roll=%4.3f Pitch=%4.3f Yaw=%4.3f"%(tvec[0],tvec[1],tvec[2],roll*(180/3.14),pitch*(180/3.14),yaw*(180/3.14)))
        #return (tvec[0],tvec[1],tvec[2],roll,pitch,yaw)


        # get camera position wrt to marker
        pos_camera = -R_tc*np.matrix(tvec).T
        roll,pitch,yaw = rotationMatrixToEulerAngles(R_flip*R_tc)
        print("X=%4.3f Y=%4.3f Z=%4.3f Roll=%4.3f Pitch=%4.3f Yaw=%4.3f Distance=%4.3f"%(pos_camera[0],pos_camera[1],pos_camera[2],
                    roll*(180/3.14),pitch*(180/3.14),yaw*(180/3.14),
                    getDistanceToMarker(pos_camera[0],pos_camera[1],pos_camera[2])))
        return (pos_camera[0],pos_camera[1],pos_camera[2],
                roll,pitch,yaw)
    else:
        print("No markers found!")
        return None

def detect_markers(img):
        markerLength = 100
        aruco_list = []
        camera_matrix,dist_coeff = getCameraMatrix()
	######################## INSERT CODE HERE ########################
        j=0
        aruco_dict=aruco.Dictionary_get(aruco.DICT_5X5_250)
        parameters=aruco.DetectorParameters_create()
        #img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        corners,ids,_=aruco.detectMarkers(img,aruco_dict,parameters=parameters)
        for i in corners:
                id_cur=ids[j]
                j+=1
                rvec, tvec, _= aruco.estimatePoseSingleMarkers(i,100,camera_matrix,dist_coeff)
                centerX=0
                centerY=0
                for x,y in i[0]:
                        centerX+=x
                        centerY+=y
                centerX/=4
                centerY/=4
                aruco_list.append((id_cur,(centerX,centerY),rvec,tvec))
                #print (aruco_list)
	##################################################################
        return aruco_list


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret = getCameraPosition(gray,expectedId,frame=frame)
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break