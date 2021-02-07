import numpy as np
from math import sin,cos,tan

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
