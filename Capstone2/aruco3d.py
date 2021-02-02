import open3d as o3d
import numpy as np
import numpy as np
from math import sqrt, sin, cos, tan, atan
from detect import getCameraPosition
from matplotlib import pyplot as plt
import cv2

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


def translatePoint(x, y, z):
    '''
        Makes the translation matrix
    '''
    translationMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [x, y, z, 1],
    ])
    return translationMatrix


def translate(x, y, z, points):
    '''
        translate a given set of points about x,y,z axis
    '''
    translationMatrix = translatePoint(x, y, z)
    out = list(map(lambda point: np.matmul(np.append(point, [1.0]), translationMatrix)[:-1], points))
    out = np.array(out)
    return out


def makeAxis(scale=50):
    '''
        Makes the coordinates for the axis
    '''
    coordinates = []
    test = np.arange(0.0, scale, 0.25)
    for i in test:
        coordinates.append([0.0, 0.0, i])
        coordinates.append([i, 0.0, 0.0])
        coordinates.append([0.0, i, 0.0])
        coordinates.append([0.0, 0.0, -i])
        coordinates.append([-i, 0.0, 0.0])
        coordinates.append([0.0, -i, 0.0])
    coordinates = np.array(coordinates)
    return coordinates


def draw(objects=None, others=None, axis=True):
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

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

p1 = getCameraPosition(gray1, 6, frame=img1, marker_size=9.5)
p2 = getCameraPosition(gray2, 6, frame=img2, marker_size=9.5)

'''
plt.imshow(img1)
plt.show()

plt.imshow(img2)
plt.show()
'''

camDict = {
    "f": 2.2,
    "view": (57, 43),
    "res": (640, 480)
}


# There is no cock like horse cock ///♫♫♫♫♫♫♫♫♫♫♫♫♫♫
# Send your asshole into shock /// ♫♫♫♫♫♫♫


def camera2canvas(camStuff):
    xAngleRange = np.arange(-camStuff["view"][0] / 2, camStuff["view"][0] / 2,
                            camStuff["view"][0] / camStuff["res"][0])
    yAngleRange = np.arange(-camStuff["view"][1] / 2, camStuff["view"][1] / 2,
                            camStuff["view"][1] / camStuff["res"][1])
    xAngleRange = xAngleRange * (3.1415 / 180)
    yAngleRange = yAngleRange * (3.1415 / 180)
    coordWRTC = np.zeros((camStuff["res"][0] * camStuff["res"][1], 3))
    for i in range(camStuff["res"][0]):
        for j in range(camStuff["res"][1]):
            coordWRTC[i * camStuff["res"][1] + j] = np.array([
                camStuff["f"] * tan(xAngleRange[i]),
                camStuff["f"] * tan(xAngleRange[j]),
                -camStuff["f"]
            ])
    return coordWRTC


def ThreeDPixelsToPoints(camStuff, pixels, R, t):
    """

    for x,y in img1:
        x,y,z -> x1,y1,z1

    """
    points = []
    for x, y in pixels:
        alpha = (x - camStuff["res"][0]/2) * (camStuff["view"][0] * (pi / 180) / camStuff["res"][0])
        beta = (y - camStuff["res"][1]/2) * (camStuff["view"][1] * (pi / 180) / camStuff["res"][1])
        point = np.array([camStuff["f"] * tan(alpha), camStuff["f"] * tan(beta), camStuff["f"]])
        points.append(point)
    points = np.array(points)
    points = translate(t[0], t[1], t[2], points)
    points = rotate(R[0], R[1], R[2], points)
    return points


def ThreeDPointsToPixels(camStuff, points, R, t):
    """
    
    for x,y in img1:
        x,y,z -> x1,y1,z1

    """
    pixels = []

    points = urotate(R[0], R[1], R[2], points)
    points = translate(t[0], t[1], t[2], points)
    for point in points:
        alpha = atan(point[0] / camStuff["f"])
        beta = atan(point[1] / camStuff["f"])
        x = alpha * camStuff["res"][0] / (camStuff["view"][0] * (pi / 180)) + camStuff["res"][0]/2
        y = beta * camStuff["res"][1] / (camStuff["view"][1] * (pi / 180)) + camStuff["res"][1]/2
        pixels.append((x, y))
    return np.array(pixels)


c = camera2canvas(camDict)


def calMatrices(pos1, pos2):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    diff = pos1 - pos2
    t = np.array(diff[0], diff[1], diff[2])
    R = rotateMatrix(diff[3], diff[4], diff[5])
    return R, t


# p11 = makeAxis(scale=10)
# p1 = rotate(3.14/4,3.14/3,3.14/3.666,p1)

# m1 = makeAxis(scale=5)
# m1 = translate(p1[0],p1[1],p1[2],m1)
# m1 = rotate(p1[3],p1[4],p1[5],m1)
# c1 = translate(p1[0],p1[1],p1[2],c)
# c1 = rotate(p1[3],p1[4],p1[5],c1)
#
#
# m2 = makeAxis(scale=5)
# m2 = translate(p2[0],p2[1],p2[2],m2)
# m2 = rotate(p2[3],p2[4],p2[5],m2)
# c2 = translate(p2[0],p2[1],p2[2],c)
# c2 = rotate(p2[3],p2[4],p2[5],c2)


R = np.array([1.2, 2.6, 3.1])
# R = np.array([3.1415,3.1415,3.1415])
t = np.array([2, 4, 8])
#
a = np.array([
    (200, 400),
    (500, 300),
    (69, 69)
])
#
# b = np.array([
#     np.array([1, 2, 3]),
#     np.array([4.20, 6.9, 0.0]),
#     np.array([1, 1, 1])
# ])
#
# aa = rotate(R[0], R[1], R[2], b)
# print(aa)
# bb = rotate(-R[0], -R[1], -R[2], aa)
# print(bb)
dd = ThreeDPixelsToPoints(camStuff=camDict, pixels=a, R=R, t=t)
print(dd)
ee = ThreeDPointsToPixels(camStuff=camDict, points=dd, R=(-1 * R), t=(-1 * t))
print(ee)

# draw(objects=[p11,m1,m2,c1,c2],axis=False)

#
# lol = rotateMatrix(0, 0, pi/2)
# lol2 = np.matmul(lol, np.array([1, 0, 0]))
# lol3 = np.matmul(np.identity(3), np.array([1, 0, 0]))
# print(lol2)
# print(lol3)

# a = 1.1
# b = 2.3
# c = 3.1
#
# lol = np.matmul(rotateMatrix(a, b, c), ultaRotateMatrix(-1*a, -1*b, -1*c))
# print(lol)
