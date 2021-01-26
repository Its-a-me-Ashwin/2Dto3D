from detect import getCameraPosition,getDistanceToMarker
import cv2
import numpy as np
from math import sqrt,sin,cos,tan


convert = 3.1415/180.0

# measured in (cm) 
# angles in radians
camDict = {
           "f" : 20,
           "view":(57,43),
           "res" : (640,480)
            }


def rotatePoint(x,y,z):
    rotations = np.array([
        [cos(x)*cos(y),cos(x)*sin(y)*sin(z)-sin(x)*cos(z),cos(x)*sin(y)*cos(z)+sin(x)*sin(z)],
        [sin(x)*cos(y),sin(x)*sin(y)*sin(z)+cos(x)*cos(z),sin(x)*sin(y)*cos(z)-cos(x)*sin(z)],
        [-sin(y),cos(y)*sin(z),cos(y)*cos(z)]
    ])
    return rotations

def rotate(x,y,z,points):
    out = list()
    rotationalMatrix = rotatePoint(x,y,z)
    for i in range(points.shape[0]):
        out.append(np.matmul(rotationalMatrix,points[i]))
    out = np.array(out)
    return out

def translatePoint(x,y,z):
    '''
        Makes the translation matrix
    '''
    translationMatrix = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [x,y,z,1],
        ])
    return translationMatrix

def translate(x,y,z,points):
    '''
        translate a given set of points about x,y,z axis
    '''
    translationMatrix = translatePoint(x,y,z)
    out = list(map(lambda point:np.matmul(np.append(point,[1.0]),translationMatrix)[:-1],points))
    out = np.array(out)
    return out


def makeRTMatrices(data1,data2):
    data = np.array(data2)-np.array(data1)
    T = np.array([data[0],data[1],data[2]]).reshape((1,3))[0]
    T = np.array(
        [
            [0.0,-T[2],T[1]],
            [T[2],0.0,-T[0]],
            [-T[1],T[0],0.0]
        ])
    R = rotatePoint(data[3],data[4],data[5])
    return T,R

def makeEsentialMatrix(data1,data2):
    data = np.array(data2)-np.array(data1)
    T = np.array([data[0],data[1],data[2]]).reshape((1,3))[0]
    T = np.array(
        [
            [0.0,-T[2],T[1]],
            [T[2],0.0,-T[0]],
            [-T[1],T[0],0.0]
        ])
    R = rotatePoint(data[3],data[4],data[5])
    if not (np.all(abs(np.matmul(R,np.transpose(R))-np.identity(3,dtype=R.dtype))<1e-6)):
        print("Error")
        return
    return np.matmul(R,T)

def camera2canvas(camStuff):
    xAngleRange = np.arange(-camStuff["view"][0]/2,camStuff["view"][0]/2,
                    camStuff["view"][0]/camStuff["res"][0])
    yAngleRange = np.arange(-camStuff["view"][1]/2,camStuff["view"][1]/2,
                    camStuff["view"][1]/camStuff["res"][1])
    xAngleRange = xAngleRange * (3.1415/180)
    yAngleRange = yAngleRange * (3.1415/180)
    coordWRTC = np.zeros((camStuff["res"][0]*camStuff["res"][1],3))
    for i in range(camStuff["res"][0]):
        for j in range(camStuff["res"][1]):
            coordWRTC[i*camStuff["res"][1]+j] = np.array([
                camStuff["f"]*tan(xAngleRange[i]),
                camStuff["f"]*tan(xAngleRange[j]),
                camStuff["f"]
                ])
    return coordWRTC    



p1 = '1.jpg'
p2 = '2.jpg'

img1 = cv2.imread(p1,0)
img2 = cv2.imread(p2,0)

# get canvas coordinates
coordWRTC = camera2canvas(camDict)

# get camera position
ret1 = getCameraPosition(img1,6,13.5)
ret2 = getCameraPosition(img2,6,13.5)

# get Essential Matrix
#E = makeEsentialMatrix(ret1,ret2)

# get projected coordinates

coordWRTC1 = rotate(ret1[3],ret1[4],ret1[5],coordWRTC)
coordWRTC1 = translate(ret1[0],ret1[1],ret1[2],coordWRTC)


coordWRTC2 = rotate(ret2[3],ret2[4],ret2[5],coordWRTC)
coordWRTC2 = translate(ret2[0],ret2[1],ret2[2],coordWRTC)



#for i in range(img1.shape[0]):
#    for j in range(img1.shape[1]):
