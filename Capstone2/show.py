import open3d as o3d
import numpy as np
from detect import getCameraPosition,getDistanceToMarker
import cv2
import numpy as np
from parallax import camera2canvas,translate
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
    ],dtype=np.float64)
    return rotations

def rotate(x,y,z,points):
    out = list()
    rotationalMatrix = rotatePoint(x,y,z)
    for i in range(points.shape[0]):
        out.append(np.matmul(rotationalMatrix,points[i]))
    out = np.array(out)
    return out



def makeAxis(scale=50):
    '''
        Makes the coordinates for the axis
    '''
    coordinates = []
    test = np.arange(0.0,scale,0.25)
    for i in test:
        coordinates.append([0.0,0.0,i])
        coordinates.append([i,0.0,0.0])
        coordinates.append([0.0,i,0.0])
        coordinates.append([0.0,0.0,-i])
        coordinates.append([-i,0.0,0.0])
        coordinates.append([0.0,-i,0.0])
    coordinates = np.array(coordinates)    
    return coordinates


def makeBox (x,y,z,r,p,ya,scale=10):
    coordinates = []
    test = np.arange(0.0,scale,0.25)
    for i in test:
        coordinates.append([x,y,z+i])
        coordinates.append([x+i,y,z])
        coordinates.append([x,y+i,z])
        coordinates.append([x,y,-i])
        coordinates.append([x-i,y,z])
        coordinates.append([x,y-i,z])
    coordinates = np.array(coordinates,dtype=np.float64)
    coordinates = rotate(r,p,ya,coordinates)    
    return coordinates


def draw(objects = None,axis = True):
    pointCloud = list()
    if object != None:
        for idx in range(len(objects)):
            pcd0 = o3d.geometry.PointCloud()
            pcd0.points = o3d.utility.Vector3dVector(objects[idx])
            pointCloud.append(pcd0)
    if axis:
        c = makeAxis()
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(c)
        
        pointCloud.append(pcd1)
    print(pointCloud)
    o3d.visualization.draw_geometries(pointCloud)



p1 = '1.jpg'
p2 = '2.jpg'

img1 = cv2.imread(p1,0)
img2 = cv2.imread(p2,0)

ret1 = getCameraPosition(img1,6,frame=img1,marker_size=9.5)
ret2 = getCameraPosition(img2,6,frame=img2,marker_size=9.5)

#ret1 = getCameraPosition(img1,6,13.5)
#ret2 = getCameraPosition(img2,6,13.5)

c1 = makeBox(ret1[0],ret1[1],ret1[2],ret1[3],ret1[4],ret1[5])
c2 = makeBox(ret2[0],ret2[1],ret2[2],ret2[3],ret2[4],ret2[5])


#coordWRTC = camera2canvas(camDict)

draw([c1,c2])