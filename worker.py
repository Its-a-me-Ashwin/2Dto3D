# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:33:23 2020

@author: 91948
"""
import os
import cv2
#mat = scipy.io.loadmat('nyu_depth_v2_labeled.mat')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import math
import open3d as o3d
from numpy import sin,cos,tan
from tqdm import tqdm
from keras.models import load_model
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as backend
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
import json
from lol.utils.arucolib import ArucoSingleTracker
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
    out = list(map(lambda point:np.matmul(point,rotationalMatrix),points))
    out = np.array(out)
    return out

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

def makeAxis(scale=2.5):
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


FX = 5.8262448167737955e+02
FY = 5.8269103270988637e+02
CX = 3.1304475870804731e+02
CY = 2.3844389626620386e+02
# DP1 = 351.3
# DP2 = 1092.5
'''FX = 300
FY = 300
CX = 320
CY = 240
mm = lm("lolgfull.h5", custom_objects={'berhu_loss':berhu_loss, 'rmse':rmse})

def give_password(name):
    a = [ord(i) for i in name]
    for i in range(len(a)*10):
        a[i%len(a)] = (a[(i)%len(a)] + a[(i+1)%len(a)])%26 + 97
    return "".join([chr(i) for i in a])


def read_nyu_pgm(filename, byteorder='>'):
    """
        read pgm depth
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, _ = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out


def create_xyz(img, dpt, edg, k = 2):
    """
    """
    edge = np.zeros((dpt.shape[0]+2*k,dpt.shape[1]+2*k))
    edge[k:dpt.shape[0]+k, k:dpt.shape[1]+k] = edg
    cloud = np.zeros((dpt.shape[0]*dpt.shape[1], 3))
    color = np.zeros((dpt.shape[0]*dpt.shape[1], 3))
    a = 0
    b = 0
    c = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # # From c++ point cloud thingy
            # d = dpt[i][j]
            # xx = DP1/(DP2 - d*(2**8))
            # if xx <= 0:
            #     dpt[i][j] = 0
            # else:
            #     dpt[i][j] = xx*1000 + 0.5
            
            # if (edg[i-k:i+k, j-k:j+k] == 0).all():
            a = (i-CX)*dpt[i][j]/FX
            b = (j-CY)*dpt[i][j]/FY
            c = dpt[i][j]
            cloud[i*dpt.shape[1]+j][0] = a
            cloud[i*dpt.shape[1]+j][1] = b
            cloud[i*dpt.shape[1]+j][2] = c
            color[i*dpt.shape[1]+j] = img[i][j]/255.0

    return cloud, color
'''
def render (depth,rgb,axis=True,draw=False):
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
    if draw:
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
    

def draw(objects,axis = False):
    pointCloud = list()
    for idx in range(len(objects)):
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(objects[idx]['points'])
        pcd0.colors = o3d.utility.Vector3dVector(objects[idx]['colors'])
        pointCloud.append(pcd0)
    if axis:
        c = makeAxis()
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(c)
        
        pointCloud.append(pcd1)
    o3d.visualization.draw_geometries(pointCloud)


def load(axis=True):
    '''
        Load and render test files from model
    '''
    # Change these file paths
    depth = np.load('0_0DEPTH.npy')
    img = np.load('0_0IMGS.npy')
    #print(img.shape,depth.shape)
    for idx in range(3):
        imgA = cv2.resize(img[idx],(640,480))
        imgA = np.array(imgA*255.0,dtype=np.uint8)
        
        depthA = cv2.resize(depth[idx],(640,480))
        depthA[:,:,2] = (depthA[:,:,2]*0.5+0.5)+2.0
        depthA = (depthA*10).astype(np.float64)
        
        colors = np.zeros((imgA.shape[0]*imgA.shape[1],3))
        points = np.zeros((depthA.shape[0]*depthA.shape[1],3))
        for i in range(depthA.shape[0]):
            for j in range(depthA.shape[1]):
                #points[i*depthA.shape[1]+j] = depthA[i][j]
                colors[i*imgA.shape[1]+j] = imgA[i][j]/255.0
        
        points[:,0] = np.reshape(depthA[:,:,0],(depthA.shape[0]*depthA.shape[1],))
        points[:,1] = np.reshape(depthA[:,:,1],(depthA.shape[0]*depthA.shape[1],))
        points[:,2] = np.reshape(depthA[:,:,2],(depthA.shape[0]*depthA.shape[1],))
        
        plt.imshow(imgA)
        plt.show()
        pointCloud = []
        if axis:
            c = makeAxis()
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(c)
            pointCloud.append(pcd1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pointCloud.append(pcd)
        #o3d.io.write_point_cloud("sync.ply", pcd)
        o3d.visualization.draw_geometries(pointCloud)
    
    
def makeMatrix (angleX=60,angleY=60,shape=(480,640)): 
    '''
        Constant Matrix
        Set 60,60 for model
    '''
    kinectX = angleX#60 # X view angle 53
    kinectY = angleY#60 # Y view angle 43
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

def fixRes(cameraAngles,cameraResolution):
    '''
        Fix resolution and angle distortions
    '''
    # default model
    modelSpecific = makeMatrix()
    # current camera
    cameraSpecific = makeMatrix(cameraAngles[0],cameraAngles[1],shape=cameraResolution)
    return cameraSpecific/modelSpecific

    
def project (model,imgs,angles,cameraAngles=(53,43), cameraResolution=(480,640)):
    offsets = fixRes(cameraAngles, cameraResolution)
    imgs = imgs/255.0
    imgs = (imgs-0.5)*2.0
    predictions = model.predict(imgs)
    pcds = list()
    for idx in range(len(imgs)):
        temp = cv2.resize(predictions[idx],(640,480))
        temp[:,:,0] = temp[:,:,0]*offsets[:,:,0]
        temp[:,:,1] = temp[:,:,1]*offsets[:,:,1]
        temp = np.reshape(temp,(480*640,3))
        temp = rotate(angles[idx][0],angles[idx][1],angles[idx][2],temp)
        color = np.reshape(cv2.resize(imgs[idx],(640,480)),(640*480,3))/255.0
        #print(color.shape)
        #print(temp.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(temp)
        pcd.colors = o3d.utility.Vector3dVector(color)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)




def predictNN(images, model,show=False):
    imgs = list()
    shapes = list()
    for im in images:
        imgs.append(cv2.resize(im,(480,640)))
        shapes.append((im.shape[0],im.shape[1]))
    imgs = np.array(imgs)
    imgs = imgs/127.5 -1.0
    predictions = model.predict(imgs)
    predictions = (predictions*0.5+0.5)
    predictions = predictions*(-1)
    depths = list()
    #print(predictions.shape)
    for idx in range(len(shapes)):
        #depths.append(cv2.resize(np.reshape(predictions[idx],(256,256)),(shapes[idx][1], shapes[idx][0])))
        depths.append(cv2.resize(predictions[idx],(480,640)))
        #depths.append(cv2.resize(predictions[idx],(640,480)))
        if show:
            plt.imshow(depths[-1])
            plt.show()
    return depths



def makeJSONoffsets (path):
    id_to_find  = 6
    marker_size  = 9 #- [cm]

    #--- Get the camera calibration path
    calib_path  = ""
    camera_matrix   = np.loadtxt(calib_path+'cameraMatrix_bck.txt', delimiter=',')
    camera_distortion   = np.loadtxt(calib_path+'cameraDistortion_bck.txt', delimiter=',')

    aruco_tracker = ArucoSingleTracker(id_to_find=id_to_find, marker_size=marker_size, camera_size=[640,480], show_video=True, camera_matrix=camera_matrix, camera_distortion=camera_distortion)

    detections = dict()
    for imgFile in tqdm(os.listdir(path)):
        try:
            gayPath = os.path.join(path,imgFile)
            if gayPath.endswith('json'): continue
            img = cv2.imread(gayPath)
            #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            detection = aruco_tracker.track2(img)
            detections[imgFile] = detection["cp"]+list(map(lambda x: x*(180/3.1415),detection["ca"]))
        except Exception as e:
            print(e,end="\n\n\n\n\n\n\n\n\n\n\n\n\n")
    #for k,v in detections.items():
    #    detections[k] = v["cp"]+list(map(lambda x: x*(180/3.1415),v["ca"]))
    #offsets = list(map(lambda x: x["cp"]+x["ca"],detections))
    print(detections)
    jsonData = json.dumps(detections)
    with open(os.path.join(path,'offsets.json'),"w") as outFile:
        outFile.write(jsonData)
    return "Go fuck ur self"
    


def work (model,images,offsets):
    '''
    Parameters
    ----------
    images : TYPE
        DESCRIPTION.
    offsets : [(xT,yT,zT,xR,yR,zR),....11] radians and meters
        DESCRIPTION.

    Returns
    -------
    yo mama
    '''
    depths = predictNN(images,model)
    objects = list()
    n_images = len(images)
    c = 0
    for image,depth,offset in zip(images,depths,offsets):
        #print(image.shape, depth.shape)
        points,colors = render(depth,image)
        plt.subplot(n_images, 2, c*2+1)
        plt.title('image')
        plt.imshow(image)
        plt.subplot(n_images, 2, c*2+2)
        plt.title('depth')
        plt.imshow(depth)
        
        points[:,2] = points[:,2] - points[:,2].mean()
        points = rotate(offset[3],offset[4],offset[5],points)
        points = translate(offset[0],offset[1],offset[2],points)
        
        #points = translate(offset[0],offset[1],offset[2]+(depth.mean()),points)
        #plt.show()
        
        objects.append({'points':points,'colors':colors})
        c+=1
    plt.savefig('plot.png', bbox_inches='tight')
    draw(objects)
    return objects
                
    
def readFolder(path, model,resize=True):
    makeJSONoffsets(path) # remove this to revert back to old 
    f = open(os.path.join(path,'offsets.json'),'r') 
    data = json.load(f)
    f.close()
    images = list()
    offsets = list()
    for imgFile,offset in data.items():
        img = cv2.imread(os.path.join(path,imgFile))
        #img = cv2.cvtColor(img,cv2.COLOR_BAYER_BGR2RGB)
        #print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            #img = cv2.resize(img,(480,640))
            #plt.imshow(img)
            #plt.show()
            img = cv2.resize(img,(480,640))
            #plt.imshow(img)
            #plt.show()
        images.append(img)
        offsets.append(np.array(offset)*(3.14/180))
    return work(model, images, offsets)

if __name__ == '__main__':
    #makeJSONoffsets('./test2')
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
    model = load_model('nyu.h5', custom_objects=custom_objects, compile=False)
    readFolder('./test2',model)
    '''
    gan = Pix2Pix()
    gan.generator.load_weights("lolPoint.h5")
    imgs = list()
    offsets = list()
    off = 0
    for i in range(4):
        imgs.append(cv2.resize(cv2.imread('420.png'),(256,256)))
        offsets.append((off,0,0))
        off += 3.1415/2
    offsets = np.array(offsets)
    imgs = np.array(imgs)
    project(gan.generator,imgs,offsets)
    '''
    if False:
        with h5py.File(path, 'r') as f:
            data = f.keys()
            #print(data)
            idx=420
            depth = f['depths'][idx]
            image = f['images'][idx]
            #print(depth.shape)
            lll = [375]
            imgs = []
            for idx in lll:
                image = f['images'][idx]
                image = np.rollaxis(image,0,3)
                imgs.append(cv2.cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE))
                
            ofs = [[0,0,0,0,0,0], [0,0,0,0,0,1.5707963268], [0,0,0,0,0,1.5707963268*2], [0,0,0,0,0,1.5707963268*3]]
            work(imgs, ofs)
        '''
        xyz,colors = render(depth,image,draw=False)
        xyz1 = translate(3,3,3,xyz)
        xyz1 = rotate(0,3.14,0,xyz1)
        objects = list()
        objects.append(dict())
        objects[0]['points'] = xyz
        objects[0]['colors'] = colors
        
        objects.append(dict())
        objects[1]['points'] = xyz1
        objects[1]['colors'] = colors
        
        draw(objects,axis=True)
    
    readFolder('./Test2')'''
