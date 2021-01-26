import open3d as o3d
import numpy as np
import numpy as np
from math import sqrt,sin,cos,tan
from detect import getCameraPosition
from matplotlib import pyplot as plt
import cv2





def rotateMatrix(z,y,x):
    '''
    Point (n,3)
    radians
    '''
    rotationMatrix = np.array([
        [cos(x)*cos(y),cos(x)*sin(y)*sin(z)-sin(x)*cos(z),cos(x)*sin(y)*cos(z)+sin(x)*sin(z)],
        [sin(x)*cos(y),sin(x)*sin(y)*sin(z)+cos(x)*cos(z),sin(x)*sin(y)*cos(z)-cos(x)*sin(z)],
        [-sin(y),cos(y)*sin(z),cos(y)*cos(z)]
    ],dtype = np.float64)
    # (n,3) * (3*3) -> (n,3)
    return rotationMatrix


def rotate(x,y,z,points):
    '''
    Point (n,3)
    radians
    '''
    rotationMatrix = rotateMatrix(x,y,z)
    # (n,3) * (3*3) -> (n,3)
    convertedPoints = np.matmul(points,rotationMatrix)
    return convertedPoints



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



def draw(objects = None,others=None,axis = True):
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
    if others != None:
        pointCloud.extend(others)
    o3d.visualization.draw_geometries(pointCloud)


img1 = cv2.imread('5.jpg')
img2 = cv2.imread('6.jpg')

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

p1 = getCameraPosition(gray1,6,frame=img1,marker_size=9.5)
p2 = getCameraPosition(gray2,6,frame=img2,marker_size=9.5)


'''
plt.imshow(img1)
plt.show()

plt.imshow(img2)
plt.show()
'''

camDict = {
           "f" : 2.2,
           "view":(57,43),
           "res" : (640,480)
            }

#There is no cock like horse cock ///♫♫♫♫♫♫♫♫♫♫♫♫♫♫
#Send your asshole into shock /// ♫♫♫♫♫♫♫


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
                -camStuff["f"]
                ])
    return coordWRTC  


def ThreeDPointToPixel(camPoint, camStuff, point):
    """
    
    for x,y in img1:
        x,y,z -> x1,y1,z1

    """
    return



c = camera2canvas(camDict)

def calMatrices(pos1,pos2):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    diff = pos1-pos2
    t = np.array(diff[0], diff[1], diff[2])
    R = rotateMatrix(diff[3], diff[4], diff[5])
    return R, t


p11 = makeAxis(scale=10)
#p1 = rotate(3.14/4,3.14/3,3.14/3.666,p1)

m1 = makeAxis(scale=5)
m1 = translate(p1[0],p1[1],p1[2],m1)
m1 = rotate(p1[3],p1[4],p1[5],m1)
c1 = translate(p1[0],p1[1],p1[2],c)
c1 = rotate(p1[3],p1[4],p1[5],c1)


m2 = makeAxis(scale=5)
m2 = translate(p2[0],p2[1],p2[2],m2)
m2 = rotate(p2[3],p2[4],p2[5],m2)
c2 = translate(p2[0],p2[1],p2[2],c)
c2 = rotate(p2[3],p2[4],p2[5],c2)


draw(objects=[p11,m1,m2,c1,c2],axis=False)
