# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:10:03 2020

@author: 91948
"""
import scipy.io
#mat = scipy.io.loadmat('nyu_depth_v2_labeled.mat')
import matplotlib.pyplot as plt
path = 'nyu_depth_v2_labeled.mat'
import h5py
import numpy as np

'''
with h5py.File(path, 'r') as f:
    data = f.keys()
    print(data)
    Z = f['rawDepths'][0]
    print(Z)
    K = list()
    for i in range(Z.shape[0]):
        K.append([])
        for j in range(Z.shape[1]):
            K[i].append([Z[i][j],Z[i][j],Z[i][j]])
    K = np.array(K)
    print(K.shape)
    #D = np.reshape(Z,(640,480,1))
    plt.imshow(K)
'''


idx = 0
with h5py.File(path, 'r') as f:
    data = f.keys()
    print(data)
    for idx in range(2):
        Z = f['depths'][idx]
        img = f['images'][idx]
        img = np.rollaxis(img,0,3)
        labels = f['labels'][idx]
        M = Z.max()
        m = Z.min()
        K = 1.0-((Z-m)/(M-m))
        plt.imshow(K)
        plt.show()
        plt.imshow(img)
        plt.show()
        plt.imshow(labels)
        plt.show()