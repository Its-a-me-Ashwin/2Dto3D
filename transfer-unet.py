import numpy as np
import sys
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import os
import random
import cv2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model,Sequential
from tensorflow.keras.models import save_model
from sklearn.metrics import classification_report
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, Activation, Flatten

if not 'image_model' in globals():
    image_model = Xception(include_top=True, weights='imagenet')
    #image_model.summary()
    transfer_layer = image_model.get_layer('avg_pool')
    image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)

    img_size = K.int_shape(image_model.input)[1:3]
    print('img size:',img_size)



    transfer_values_size = K.int_shape(transfer_layer.output)[1]
    print('Output tensor',transfer_values_size)

#image_model_transfer.summary()
img = np.zeros(shape=(299,299,3),dtype=np.uint8)
out = image_model_transfer.predict(np.array([img]))
print(out.shape)

