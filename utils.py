# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:33:23 2020

@author: 91948
"""
import cv2
#mat = scipy.io.loadmat('nyu_depth_v2_labeled.mat')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import math
import open3d as o3d
from numpy import sin,cos,tan
#from tensorflow.keras.models import load_model
PI = math.pi
idx = 0
path = 'nyu_depth_v2_labeled.mat'


def rotatePoint(x,y,z):
    '''
        Makes the ratation matrix
    '''
    rotationalMatrix = np.array([
        [cos(x)*cos(y),cos(x)*sin(y)*sin(z)-sin(x)*cos(z),cos(x)*sin(y)*cos(z)+sin(x)*sin(z)],
        [sin(x)*cos(y),sin(x)*sin(y)*sin(z)+cos(x)*cos(z),sin(x)*sin(y)*cos(z)-cos(x)*sin(z)],
        [-sin(y),cos(y)*sin(z),cos(y)*cos(z)]]
    )
    return rotationalMatrix

def rotate(x,y,z,points):
    '''
        Rotate a given set of points about x,y,z axis
    '''
    rotationalMatrix = rotatePoint(x,y,z)
    out = list(map(lambda point:np.matmul(rotationalMatrix,point),points))
    out = np.array(out)
    return out

def makeAxis(scale=10.0):
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

def makeCoordinates (depth,rgb,dim=(256,256)):
    '''
        Function for data Loader
    '''
    kinectX = 60 # X view angle 53
    kinectY = 60 # Y view angle 43
    horizAngle = kinectX*(math.pi/180.0)
    verAngle = kinectY*(math.pi/180.0)
    horizAlpha = (math.pi-horizAngle)/2
    verAlpha = (2*math.pi-verAngle/2)
    xCenter = depth.shape[0]//2
    yCenter = depth.shape[1]//2
    cloud = np.zeros((depth.shape[0],depth.shape[1],3))
    #colors = np.zeros((depth.shape[0]*depth.shape[1],3))
    if rgb.shape[0] == 3:
        rgb = np.rollaxis(rgb,0,3)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            cloud[i][j][0] = depth[i][j]/math.tan(horizAlpha + (i*horizAngle)/depth.shape[0]) # X ordinate
            cloud[i][j][1] = depth[i][j]*math.tan(verAlpha + (j*verAngle)/depth.shape[1]) # Y ordinae
            cloud[i][j][2] = depth[i][j]
            #colors[i*depth.shape[1]+j] = rgb[i][j]/255.0
    #X = cloud[:,0]
    #Y = cloud[:,1]
    #Z = cloud[:,2]
    #xyz = np.dstack((X,Y,Z))[0]
    cloud = cv2.resize(cloud,dim)
    return cloud


 

def render (depth,rgb,axis=True):
    '''
        Render test points from NYU dataset
    '''
    kinectX = 53 # X view angle 53
    kinectY = 43 # Y view angle 43
    horizAngle = kinectX*(math.pi/180.0)
    verAngle = kinectY*(math.pi/180.0)
    horizAlpha = (math.pi-horizAngle)/2
    verAlpha = (2*math.pi-verAngle/2)
    xCenter = depth.shape[0]//2
    yCenter = depth.shape[1]//2
    cloud = np.zeros((depth.shape[0]*depth.shape[1],3))
    colors = np.zeros((depth.shape[0]*depth.shape[1],3))
    if rgb.shape[0] == 3:
        rgb = np.rollaxis(rgb,0,3)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            cloud[i*depth.shape[1]+j][0] = depth[i][j]/math.tan(horizAlpha + (i*horizAngle)/depth.shape[0]) # X ordinate
            cloud[i*depth.shape[1]+j][1] = depth[i][j]*math.tan(verAlpha + (j*verAngle)/depth.shape[1]) # Y ordinae
            cloud[i*depth.shape[1]+j][2] = depth[i][j]+2.0
            colors[i*depth.shape[1]+j] = rgb[i][j]/255.0
    X = cloud[:,0]
    Y = cloud[:,1]
    Z = cloud[:,2]
    xyz = np.dstack((X,Y,Z))[0]
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(xyz*2)
    pcd0.colors = o3d.utility.Vector3dVector(colors)
    pointCloud = list()
    pointCloud.append(pcd0)
    if axis:
        c = makeAxis()
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(c)
        pointCloud.append(pcd1)
    o3d.visualization.draw_geometries(pointCloud)
    return xyz,colors
    
def load(axis=True):
    '''
        Load and render test files from model
    '''
    # Change these file paths
    depth = np.load('0_500DEPTH.npy')
    img = np.load('0_500IMGS.npy')
    print(img.shape,depth.shape)
    for idx in range(img.shape[0]):
        imgA = cv2.resize(img[idx],(480,640))
        imgA = np.array(imgA*255.0,dtype=np.uint8)
        
        depthA = cv2.resize(depth[idx],(480,640))
        depthA[:,:,2] = depthA[:,:,2]*0.5+0.5
        depthA = np.array(depthA,dtype=np.float64)
        
        colors = np.zeros((imgA.shape[0]*imgA.shape[1],3))
        points = np.zeros((depthA.shape[0]*depthA.shape[1],3))
        for i in range(depthA.shape[0]):
            for j in range(depthA.shape[1]):
                points[i*depthA.shape[1]+j] = depthA[i][j]
                colors[i*imgA.shape[1]+j] = imgA[i][j]/255.0
        
        plt.imshow(imgA)
        plt.show()
        
        
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(points*2)
        pcd0.colors = o3d.utility.Vector3dVector(colors)
        pointCloud = list()
        pointCloud.append(pcd0)
        if axis:
            c = makeAxis()
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(c)
            pointCloud.append(pcd1)
        o3d.visualization.draw_geometries(pointCloud)
    


if __name__ == '__main__':
    with h5py.File(path, 'r') as f:
        data = f.keys()
        #print(data)
        idx=420
        depth = f['depths'][idx]
        image = f['images'][idx]
        load()
        #print(depth.shape)
        #image_print = np.rollaxis(image,0,3)
        #plt.imsave(str(idx)+'.png',image_print)
        #cloud = makeCoordinates(depth,image)
        #render(depth,image)
        #mesh = meshMaker(cloud)
        #print(xyz.shape)
        #panoroma()
