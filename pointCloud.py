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
import tensorflow as tf



kinectX = 53
kinectY = 43

horizAngle = kinectX*(math.pi/180.0)
verAngle = kinectY*(math.pi/180.0)

horizAlpha = (math.pi-horizAngle)/2
verAlpha = (2*math.pi-verAngle/2)


inData = []
outData = []
show = True
idx = 0
path = 'nyu_depth_v2_labeled.mat'
def render (depth,rgb,outputfile="sync.ply"):
    xCenter = depth.shape[0]//2
    yCenter = depth.shape[1]//2
    cloud = np.zeros((depth.shape[0]*depth.shape[1],3))
    colors = np.zeros((depth.shape[0]*depth.shape[1],3))
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
    xyz = np.dstack((X,Y,Z))[0]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(outputfile, pcd)
    
    pcd_load = o3d.io.read_point_cloud(outputfile)
    pcd_load.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_load])
    xyz_load = np.asarray(pcd_load.points)
    return pcd_load
    #return cloud,X,Y,Z
    
    
    
if False:
    with h5py.File(path, 'r') as f:
        data = f.keys()
        print(data)
        idx=420
        depth = f['depths'][idx]
        image = f['images'][idx]
        image_print = np.rollaxis(image,0,3)
        plt.imsave(str(idx)+'.png',image_print)
        cloud = render(depth,image)
        #fig = plt.figure()
        #ax = plt.axes(projection='3d')
        #ax.scatter3D(X,Z,Y, c=Z, cmap='Blues'
    
    
    


def main ():
   inputfile = '420.png'
   outputfile = '420'
   '''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   '''
   #print(">> Loading Model")
   print(">> Loading Model")
   pathDefault = 'good.h5'
   img = cv2.imread(inputfile)
   img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   origShape = img.shape
   img = cv2.resize(img,(256,256))
   img = np.reshape(img,(256,256,1))
   model = tf.keras.models.load_model(pathDefault   ) 
   print(img.shape)
   img = ((img-img.min())/(img.max()-img.min())*2.0)-1.0
   depth = model.predict(np.array([img]))
   #img = cv2.resize(img,origShape)
   #depth = cv2.resize(depth,origShape)
   d = depth[0]
   d = cv2.cvtColor(d,cv2.COLOR_GRAY2BGR)
   d = d*0.5 + 0.5
   d = (d-d.min())/(d.max()-d.min())*255.0
   d = np.array(d,dtype=np.uint8)
#    print(d.shape,d.min(),d.max())
#    print(d)
   #cv2.imwrite('gay.png',np.array((((gay+1)/2)*255),dtype=np.uint8))
   print(cv2.imwrite('depthPredict.png',d))
   
   
   
   
   
   
if __name__ == '__main__':
    print()
    main()
