import scipy
from glob import glob
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow_datasets as tfds


import random
import matplotlib.pyplot as plt
import h5py
import numpy as np
import math
from numpy import sin,cos,tan
#import preprocess
import os
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
    out = list()
    rotationalMatrix = rotatePoint(x,y,z)
    for i in range(points.shape[0]):
        out.append(np.matmul(rotationalMatrix,points[i]))
    out = np.array(out)
    return out



def makeCoordinates (depth,rgb,dim=(256,256)):
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


def imresize(a, b):
    return cv2.resize(a, b)


class DataLoader():
    def __init__(self, dataset_name, img_res=(128,128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.data_full = tfds.load('nyu_depth_v2',data_dir="/storage/tensnyu")
        self.matrix = self.makeMatrix()

    def makeMatrix (self,shape=(480,640)):
        '''
            Constant Matrix
        '''
        kinectX = 60 # X view angle 53
        kinectY = 60 # Y view angle 43
        horizAngle = kinectX*(math.pi/180.0)
        verAngle = kinectY*(math.pi/180.0)
        horizAlpha = (math.pi-horizAngle)/2
        verAlpha = (2*math.pi-verAngle/2)
        xCenter = shape[0]//2
        yCenter = shape[1]//2
        cloud = np.zeros((shape[0],shape[1],2))
        for i in range(shape[0]):
            for j in range(shape[1]):
                cloud[i][j][0] = 1/math.tan(horizAlpha + (i*horizAngle)/shape[0]) # X ordinate
                cloud[i][j][1] = math.tan(verAlpha + (j*verAngle)/shape[1]) # Y ordinae
        return cloud

    def quickCvt (self,depth,shape=(480,640)):
        '''
            Generate XYZ
        '''
        xyz = np.zeros((shape[0],shape[1],3))
        #print(xyz.shape,depth.shape,self.matrix.shape)
        xyz[:,:,0] = depth[:,:,0]*self.matrix[:,:,0]
        xyz[:,:,1] = depth[:,:,0]*self.matrix[:,:,1]
        xyz[:,:,2] = depth[:,:,0]
        return xyz
        
        
    def load_data(self, batch_size=1, is_testing=False):
        batch_data =  tfds.as_numpy(self.data_full['train'].repeat().shuffle(1024).batch(batch_size)) if not is_testing else tfds.as_numpy(self.data_full['validation'].repeat().shuffle(100).batch(batch_size))
        imgs = []
        depths = []
        one_batch = next(batch_data)
        for i in range(random.randint(0,9)):
            one_batch = next(batch_data)
        for img, depth in zip(one_batch['image'], one_batch['depth']):
            mask = (depth == 0).astype(np.uint8)
            depth = depth/10
            
            #print(depth.shape)
            depth = np.float32(depth*(256*256-1)).astype(np.uint16)#np.array(cv2.cvtColor(np.float32(dddd2*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            dst = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)
            depth = (dst / (256*256-1))
            depth = (depth - depth.min())/ (depth.max()-depth.min())
#             depth = np.reshape(depth,(480,640,1))
#             depth = self.quickCvt(depth)
#             depth[:,:,2] = depth[:,:,2]*2.0 -1.0
            img = imresize(img, self.img_res)
            depth = imresize(depth, self.img_res)
            depth = cv2.resize(depth,(256,256))
            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img = np.fliplr(img)
                depth = np.fliplr(depth)
            
            depth = np.reshape(depth, (depth.shape[0],depth.shape[0],1))
            
            imgs.append(img)
            depths.append(depth)

        imgs = np.array(imgs)/127.5 - 1.
        depths = np.array(depths)*2 - 1

        return imgs, depths

    def load_batch(self, batch_size=1, is_testing=False):
        batch_data =  tfds.as_numpy(self.data_full['train'].repeat().shuffle(1024).batch(batch_size)) if not is_testing else tfds.as_numpy(self.data_full['validation'].repeat().shuffle(100).batch(batch_size))
        
        self.n_batches = 47584//batch_size + 1
        # print(self.n_batches)
        for _n, one_batch in enumerate(batch_data):
            if _n > self.n_batches:
                break
            imgs = []
            depths = []
            for img, depth in zip(one_batch['image'], one_batch['depth']):
                mask = (depth == 0).astype(np.uint8)
                depth = depth/10
                #print(depth.shape,img.shape)
                depth = np.float32(depth*(256*256-1)).astype(np.uint16)#np.array(cv2.cvtColor(np.float32(dddd2*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
                dst = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)
                depth = (dst / (256*256-1))
#                 depth = np.reshape(depth,(480,640,1))
                # depth = cv2.cvtColor(np.float32(depth)*255, cv2.COLOR_BGR2RGB)
#                 depth = self.quickCvt(depth)
#                 depth[:,:,2] = depth[:,:,2]*2.0 -1.0
                depth = (depth - depth.min())/ (depth.max()-depth.min())
                img = imresize(img, self.img_res)
                depth = imresize(depth, self.img_res)
                
                depth = cv2.resize(depth,(256,256))
                # If training => do random flip
                if not is_testing and np.random.random() < 0.5:
                    img = np.fliplr(img)
                    depth = np.fliplr(depth)
                depth = np.reshape(depth, (depth.shape[0],depth.shape[0],1))

                imgs.append(img)
                depths.append(depth)

            imgs = np.array(imgs)/127.5 - 1.
            depths = np.array(depths)*2 - 1

            yield imgs, depths

    def imread(self, path):
        return imageio.imread(path).astype(np.float)




# import scipy.io

# import matplotlib.pyplot as plt
# path = 'datasets/nyu_depth_v2_labeled.mat'
# import h5py
# import numpy as np


# a = DataLoader("NYU")
        
# b = a.load_batch(batch_size=4, is_testing=True)


# c = next(b)

# print(c[1][0].shape)
# print(c[0][0].shape)
# print(c[1][0].shape)

# print(c[1].shape)
