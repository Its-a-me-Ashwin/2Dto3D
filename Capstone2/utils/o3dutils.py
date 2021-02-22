import numpy as np
import open3d as o3d
from math import sin,cos,tan

camDict = {
           "f" : 4.8,
           "view":(57,43),
           "res" : (4496,3000)
            }

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

def makeLine (p1,p2,steps=50):
    # this works but i did gay shit here
    try:
        if p1[0] < p2[0]:
            x = np.arange(p1[0],p2[0],abs(p2[0]-p1[0])/steps)
        else:
            x = np.arange(p1[0],p2[0],-abs(p2[0]-p1[0])/steps)
    except:
        x = np.array([0]*50)
    try:
        if p1[1] < p2[1]:
            y = np.arange(p1[1],p2[1],abs(p2[1]-p1[1])/steps)
        else:
            y = np.arange(p1[1],p2[1],-abs(p2[1]-p1[1])/steps)
    except:
        y = np.array([0]*50)
    try:
        if p1[2] < p2[2]:
            z = np.arange(p1[2],p2[2],abs(p2[2]-p1[2])/steps)
        else:
            z = np.arange(p1[2],p2[2],-abs(p2[2]-p1[2])/steps)
    except:
        z = np.array([0]*50)
    return np.column_stack((x,y,z))


def projectImage(camStuff = camDict,image = None):
    xAngleRange = np.arange(-camStuff["view"][0]/2,camStuff["view"][0]/2,
                    camStuff["view"][0]/camStuff["res"][0])
    yAngleRange = np.arange(-camStuff["view"][1]/2,camStuff["view"][1]/2,
                    camStuff["view"][1]/camStuff["res"][1])
    xAngleRange = xAngleRange * (3.1415/180)
    yAngleRange = yAngleRange * (3.1415/180)
    coordWRTC = np.zeros((camStuff["res"][0]*camStuff["res"][1],3))
    if type(image) != None:
        colors = np.zeros((camStuff["res"][0]*camStuff["res"][1],3),dtype=np.float32)
    for i in range(camStuff["res"][0]):
        for j in range(camStuff["res"][1]):
            coordWRTC[i*camStuff["res"][1]+j] = np.array([
                camStuff["f"]*tan(xAngleRange[i]),
                camStuff["f"]*tan(yAngleRange[j]),
                -camStuff["f"]
                ])
            if type(image) != None:
                colors[i*camStuff["res"][1]+j] = image[j][i]/255.0
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(coordWRTC)
    #pcd.colors = o3d.utility.Vector3dVector(colors)
    return (coordWRTC,colors)  



def draw(objects = None,others=None,axis = True):
    '''
        Object: (n,3) n 3d points in np array
        Others: pcd objects of structures 
    '''
    pointCloud = list()
    if objects != None:
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

