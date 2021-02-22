import open3d as o3d
import numpy as np
import numpy as np
from math import sqrt,sin,cos,tan
from detect import getCameraPosition,detect_markers
from matplotlib import pyplot as plt
import cv2
#from parallax import makeEsentialMatrix



pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679


def rotateMatrixX(x):
    """

    """
    rotationMatrix = np.array([
        [1, 0, 0],
        [0, cos(x), -sin(x)],
        [0, sin(x), cos(x)]
    ], dtype=np.float64)
    return rotationMatrix


def rotateMatrixY(x):
    """

    """
    rotationMatrix = np.array([
        [cos(x), 0, sin(x)],
        [0, 1, 0],
        [-sin(x), 0, cos(x)]
    ], dtype=np.float64)
    return rotationMatrix


def rotateMatrixZ(x):
    """

    """
    rotationMatrix = np.array([
        [cos(x), -sin(x), 0],
        [sin(x), cos(x), 0],
        [0, 0, 1]
    ], dtype=np.float64)
    return rotationMatrix


def rotateMatrix(x, y, z):
    '''
    Point (n,3)
    radians
    '''
    rotationMatrix = np.matmul(rotateMatrixZ(z), rotateMatrixY(y))
    rotationMatrix = np.matmul(rotationMatrix, rotateMatrixX(x))
    return rotationMatrix


def ultaRotateMatrix(x, y, z):
    '''
    Point (n,3)
    radians
    '''
    rotationMatrix = np.matmul(rotateMatrixX(x), rotateMatrixY(y))
    rotationMatrix = np.matmul(rotationMatrix, rotateMatrixZ(z))
    return rotationMatrix


def rotate(x, y, z, points):
    '''
    Point (n,3)
    radians
    '''
    rotationMatrix = rotateMatrix(x, y, z)
    # (n,3) * (3*3) -> (n,3)
    convertedPoints = np.matmul(points, rotationMatrix)
    return convertedPoints


def urotate(x, y, z, points):
    '''
    Point (n,3)
    radians
    '''
    rotationMatrix = ultaRotateMatrix(x, y, z)
    # (n,3) * (3*3) -> (n,3)
    convertedPoints = np.matmul(points, rotationMatrix)
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



def makeEsentialMatrix(R, t):
    T = np.array(
        [
            [0.0,-t[2],t[1]],
            [t[2],0.0,-t[0]],
            [-t[1],t[0],0.0]
        ])
    if not (np.all(abs(np.matmul(R,np.transpose(R))-np.identity(3,dtype=R.dtype))<1e-6)):
        print("Error")
        return
    return np.matmul(R,T)

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')

#print(img1.shape)
img1 = cv2.resize(img1,(640,480))
img2 = cv2.resize(img2,(640,480))

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

p1 = getCameraPosition(gray1,6,frame=img1,marker_size=9.5)
p2 = getCameraPosition(gray2,6,frame=img2,marker_size=9.5)


cp1 = detect_markers(gray1)
cp2 = detect_markers(gray2)


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
                camStuff["f"]*tan(yAngleRange[j]),
                -camStuff["f"]
                ])
    return coordWRTC  


def cameraCoords2Real(camStuff,points):
    #if points.shape[1] != 2:
    #    return
    xAngleRange = np.arange(-camStuff["view"][0]/2,camStuff["view"][0]/2,
                    camStuff["view"][0]/camStuff["res"][0])
    yAngleRange = np.arange(-camStuff["view"][1]/2,camStuff["view"][1]/2,
                    camStuff["view"][1]/camStuff["res"][1])
    xAngleRange = xAngleRange * (3.1415/180)
    yAngleRange = yAngleRange * (3.1415/180)
    data = list()
    for point in points:
        #print(round(point[0]),round(point[1]))
        data.append([
                camStuff["f"]*tan(xAngleRange[int(round(point[0]))]),
                camStuff["f"]*tan(yAngleRange[int(round(point[1]))]),
                -camStuff["f"]
                ])
    data = np.array(data)
    return data

def ThreeDPointToPixel(R,t, camStuff, points):
    """
    points (n,3)
    T (1,3)
    R (3,3)
    for x,y in img1:
        x,y,z -> x1,y1,z1

    """
    print(R.shape,t.shape,points.shape)
    newCoords = np.matmul(R,(-t+points).T).T

    return newCoords



c = camera2canvas(camDict)

def calMatrices(pos1,pos2):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    diff = pos1-pos2
    diff2 = pos1+pos2 - pi
    t = np.array([float(diff[0]), float(diff[1]), float(diff[2])])
    R = rotateMatrix(diff2[3], diff2[4], diff2[5])
    return R, t



# p11 = makeAxis(scale=10)
# #p1 = rotate(3.14/4,3.14/3,3.14/3.666,p1)

# m1 = makeAxis(scale=5)
# m1 = translate(p1[0],p1[1],p1[2],m1)
# m1 = rotate(p1[3],p1[4],p1[5],m1)
# c1 = translate(p1[0],p1[1],p1[2],c)
# c1 = rotate(p1[3],p1[4],p1[5],c1)


# m2 = makeAxis(scale=5)
# m2 = translate(p2[0],p2[1],p2[2],m2)
# m2 = rotate(p2[3],p2[4],p2[5],m2)
# c2 = translate(p2[0],p2[1],p2[2],c)
# c2 = rotate(p2[3],p2[4],p2[5],c2)


#draw(objects=[p11,m1,m2,c1,c2],axis=False)

R1,t1 = calMatrices(p1,p2)
R2,t2 = calMatrices(p2,p1)

# g1 = ThreeDPointToPixel(R1,t1,camDict,c1)

# g2 = ThreeDPointToPixel(R2,t2,camDict,c2)

# draw(objects=[p11,m1,m2,c1,c2,g1,g2],axis=False)

E = makeEsentialMatrix(R=R1, t=t1)

print(cp1[0][1],cp2[0][1])

arP1 = np.array(cameraCoords2Real(camDict,[cp1[0][1]]))

arP1 = rotate(p1[3],p1[3],p1[5],arP1)
arP1 = translate(p1[0],p1[1],p1[2],arP1)


arP2 = np.array(cameraCoords2Real(camDict,[cp2[0][1]]))
arP2 = rotate(p2[3],p2[3],p2[5],arP2)
arP2 = translate(p2[0],p2[1],p2[2],arP2)

print(arP1.shape,arP2.shape)




diff = np.matmul(np.matmul(arP1,E),arP2.T)
print(E,arP1,arP2)
print(diff)

