# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:28:01 2020
@author: 91948
"""


import sys,getopt
import cv2
#mat = scipy.io.loadmat('nyu_depth_v2_labeled.mat')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import math
import open3d as o3d
from numpy import sin,cos,tan
import preprocess
import os
#from tensorflow.keras.models import load_model
PI = math.pi
idx = 0
path = 'nyu_depth_v2_labeled.mat'

def rotatePoint(x,y,z):
    rotations = np.array([
        [cos(x)*cos(y),cos(x)*sin(y)*sin(z)-sin(x)*cos(z),cos(x)*sin(y)*cos(z)+sin(x)*sin(z)],
        [sin(x)*cos(y),sin(x)*sin(y)*sin(z)+cos(x)*cos(z),sin(x)*sin(y)*cos(z)-cos(x)*sin(z)],
        [-sin(y),cos(y)*sin(z),cos(y)*cos(z)]]
    )
    return rotationalMatrix

def rotate(x,y,z,points):
    out = list()
    rotationalMatrix = rotatePoint(x,y,z)
    for i in range(points.shape[0]):
        out.append(np.matmul(rotationalMatrix,points[i]))
    out = np.array(out)
    return out

def makeAxis(scale=10.0):
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

# Input: 
# depht (X,Y)
# rgb = (3,X,Y)            
def render (depth,rgb,outputfile="sync.ply"):
    kinectX = 53 # X view angle
    kinectY = 43 # Y view angle
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
            cloud[i*depth.shape[1]+j][2] = depth[i][j]
            colors[i*depth.shape[1]+j] = rgb[i][j]/255.0
    X = cloud[:,0]
    Y = cloud[:,1]
    Z = cloud[:,2]
    
    #X = (X-X.min())/(X.max()-X.min())
    #Y = (Y-Y.min())/(Y.max()-Y.min())
    #Z = (Z-Z.min())/(Z.max()-Z.min())

    coordinates = []
    test = np.arange(0.0,10.0,0.25)
    for i in test:
        coordinates.append([0.0,0.0,i])
        coordinates.append([i,0.0,0.0])
        coordinates.append([0.0,i,0.0])
        coordinates.append([0.0,0.0,-i])
        coordinates.append([-i,0.0,0.0])
        coordinates.append([0.0,-i,0.0])
    coordinates = np.array(coordinates)

    c = o3d.geometry.PointCloud()
    c.points = o3d.utility.Vector3dVector(coordinates)

    xyz = np.dstack((X,Y,Z))[0]
    rotated_xyz = np.dstack((X,Y,Z))[0]
    rotated_xyz0 = np.dstack((X,Y,Z))[0]
    rotated_xyz1 = np.dstack((X,Y,Z))[0]
    
    rotated_xyz2 = np.dstack((X,Y,Z))[0]
    rotated_xyz3 = np.dstack((X,Y,Z))[0]
    rotated_xyz4 = np.dstack((X,Y,Z))[0]
    rotated_xyz5 = np.dstack((X,Y,Z))[0]
    


    rotated_xyz = rotate(0,PI/2,0,xyz)
    rotated_xyz0 = rotate(0,-PI/2,0,xyz)
    rotated_xyz1 = rotate(PI,0,PI,xyz)
    
    rotated_xyz2 = rotate(0,PI/4,0,xyz)
    rotated_xyz3 = rotate(0,-PI/4,0,xyz)
    rotated_xyz4 = rotate(0,3*PI/4,0,xyz)
    rotated_xyz5 = rotate(0,-3*PI/4,0,xyz)
    #xyz = np.concatenate((xyz,rotated_xyz,coordinates))

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(xyz*2)
    pcd0.colors = o3d.utility.Vector3dVector(colors)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(rotated_xyz*2)
    pcd1.colors = o3d.utility.Vector3dVector(colors)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(rotated_xyz0*2)
    pcd2.colors = o3d.utility.Vector3dVector(colors)

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(rotated_xyz1*2)
    pcd3.colors = o3d.utility.Vector3dVector(colors)

    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(rotated_xyz2*2)
    pcd4.colors = o3d.utility.Vector3dVector(colors)

    pcd5 = o3d.geometry.PointCloud()
    pcd5.points = o3d.utility.Vector3dVector(rotated_xyz3*2)
    pcd5.colors = o3d.utility.Vector3dVector(colors)

    pcd6 = o3d.geometry.PointCloud()
    pcd6.points = o3d.utility.Vector3dVector(rotated_xyz4*2)
    pcd6.colors = o3d.utility.Vector3dVector(colors)

    pcd7 = o3d.geometry.PointCloud()
    pcd7.points = o3d.utility.Vector3dVector(rotated_xyz5*2)
    pcd7.colors = o3d.utility.Vector3dVector(colors)



    #o3d.io.write_point_cloud(outputfile, pcd)
    
    #pcd_load = o3d.io.read_point_cloud(outputfile)
    #pcd_load.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd0,pcd1,pcd2,pcd3,pcd4,pcd5,pcd6,pcd7,c])
    xyz_load = np.asarray(pcd0.points)
    return pcd0,xyz_load
    #return cloud,X,Y,Z
    

def newRender (depth,rgb):
    kinectX = 60 # X view angle 53
    kinectY = 60 # Y view angle 43
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
    return xyz,colors
    
def meshMaker (pcd):
    ## use ball algorithm
    pcd.estimate_normals()
    distance = pcd.compute_nearest_neighbor_distance()
    radius = np.mean(distance)*3
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    dec_mesh = mesh.simplify_quadric_decimation(100000)
    dec_mesh.remove_degenerate_triangles()
    dec_mesh.remove_duplicated_triangles()
    dec_mesh.remove_duplicated_vertices()
    dec_mesh.remove_non_manifold_edges()
    o3d.visualization.draw_geometries([mesh])
    return dec_mesh
    
    

def panoroma ():
    '''
    clouds = np.zeros((depth.shape[0]*depth.shape[1],3))
    colors = np.zeros((depth.shape[0]*depth.shape[1],3))
    offsets = np.arange(0,180,depths.shape[0])
    for idx in range(depths.shape[0]):
        temp_cloud,_ = render(depths[idx],images[idx])
    '''
    pcd = list()
    pathD = './depths/'
    pathI = './images/'

    Ds = os.listdir(pathD)
    Is = os.listdir(pathI)
    d = dict()
    try:
        for i in range(len(Ds)):
            d[Ds[i].split('-')[0]+'.jpg'] = Ds[i]
    except:
        print("gay")
    ## fix the angles
    Xrotations = [0]*len(Ds)
    Yrotations = [0]*len(Ds)
    Zrotations = [0]*len(Ds)
    Yrotations = np.arange(0.0,PI,PI/len(Ds))
    #Xrotations = np.arange(0.0,PI,PI/len(Ds))
    #Zrotations = np.arange(0.0,PI,PI/len(Ds))
    idx = 0
    for k,v in d.items():
        try:
            print(pathD+k)
            depth = cv2.imread(pathD+v)
            image = cv2.imread(pathI+k)
            #$print(depth.shape)
            depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,(640,480))
            depth = cv2.resize(depth,(640,480))
            depth = (depth-depth.min())/(depth.max()-depth.min())
            depth = -1*(1-depth).astype(np.float32)
            depth = depth*2.5 + 0.75
            xyz,colors = newRender(depth,image)
            xyz = rotate(Xrotations[idx],Yrotations[idx],Zrotations[idx],xyz)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(xyz)
            pc.colors = o3d.utility.Vector3dVector(colors)
            pcd.append(pc)
        except Exception as e:
            print("Exception ",e)
            pass
        idx +=1
        break
    axis = makeAxis()
    axisPC = o3d.geometry.PointCloud()
    axisPC.points = o3d.utility.Vector3dVector(axis)
    pcd.append(axisPC)
    o3d.visualization.draw_geometries(pcd)

    
if __name__ == '__main__':
    with h5py.File(path, 'r') as f:
        data = f.keys()
        #print(data)
        idx=420
        depth = f['depths'][idx]
        image = f['images'][idx]
        #print(depth.shape)
        image_print = np.rollaxis(image,0,3)
        plt.imsave(str(idx)+'.png',image_print)
        cloud,xyz = render(depth,image)
        #mesh = meshMaker(cloud)
        #print(xyz.shape)
        #panoroma()
