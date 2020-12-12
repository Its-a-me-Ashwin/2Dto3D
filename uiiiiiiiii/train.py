from __future__ import print_function, division
import scipy
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
# from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# from utils import combine_images
from PIL import Image
# from capsulelayer import CapsuleLayer, PrimaryCap, Length, Mask
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, Multiply
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv3D, Conv3DTranspose, Conv2DTranspose, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader_full import DataLoader
import numpy as np
import os
import imageio
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import get_default_graph
import tensorflow as tf
import keras
import gc
import numpy as np
from keras import backend as K
from skimage import data, color, exposure
from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
import os
import glob
from tqdm import tqdm
import keras
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as backend
import cv2

folder_path = '/storage/'



# from tensorflow.compat.v1.keras.backend import set_session

# config = tf.compat.v1.ConfigProto()

# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

# config.log_device_placement = True  # to log device placement (on which device the operation ran)

# sess = tf.compat.v1.Session(config=config)

# set_session(sess)


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession, Session



def rmse(y_true, y_pred):
	  return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def rmse2(y_true, y_pred):
	  return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1)) + tf.keras.losses.MAE(y_true, y_pred)



def berhu_loss(labels, predictions, scope=None):
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")
    # with tf.name_scope(scope, "berhu_loss",
    #                     (predictions, labels)) as scope:
    # predictions = tf.to_float(predictions)
    # labels = tf.to_float(labels)

    # Make sure shape do match
    # predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    # Get absolute error for each pixel in batch 
    abs_error = tf.abs(tf.subtract(predictions, labels), name='abs_error')

    # Calculate threshold c from max error
    c = 0.2 * tf.reduce_max(abs_error)

    # if, then, else
    berHu_loss = tf.where(abs_error <= c,   
                    abs_error, 
                  (tf.square(abs_error) + tf.square(c))/(2*c))
            
    loss = tf.reduce_sum(berHu_loss)

    return loss

def build_encoder2(shape=[256,256], channel=3):
    """
    

    Parameters
    ----------
    shape : TYPE, optional
        DESCRIPTION. The default is [64,64].
    batch_size : TYPE, optional
        DESCRIPTION. The default is 16.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    shape = shape + [channel]
    # img = Input(shape=(shape[0], shape[1], shape[2]))
    # img_B = Input(shape=(shape[0], shape[1], shape[2]))
    # img = Concatenate(axis=-1)([img_A, img_B])
    img = Input(shape=(shape[0], shape[1], shape[2]))
    # img2 = Input(shape=(shape[0], shape[1], 2))
    
    
    i1 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same')(img)
    i2 = LeakyReLU()(i1)
    i4 = Conv2D(filters=64, kernel_size=9, strides=2, padding='same')(i2)
    i5 = LeakyReLU(alpha=0.2)(i4)
    i6 = BatchNormalization(momentum=0.95)(i5)
    
    i7 = Reshape(target_shape=[i6.shape[1], i6.shape[2], 16 ,4])(i6)
    # print(i7.shape)
    i8 = Conv3D(filters = 64, kernel_size=(6,6,4), strides=2, padding='same')(i7)
    i9 = LeakyReLU(alpha=0.2)(i8)
    i10 = BatchNormalization(momentum=0.95)(i9)
    # print(i10.shape)
    i11 = Conv3D(filters = 128, kernel_size=(4,4,5), strides=2, padding='same')(i10)
    i12 = LeakyReLU(alpha=0.2)(i11)
    i13 = BatchNormalization(momentum=0.95)(i12)

    i14 = Conv3D(filters = 256, kernel_size=(3,3,2), strides=2, padding='same')(i13)
    i15 = LeakyReLU(alpha=0.2)(i14)
    i16 = BatchNormalization(momentum=0.95)(i15)

    
    x = Conv3DTranspose(filters = 256, kernel_size=(3,3,2), strides=2, padding='same')(i16)
    x = Dropout(0.2)(x)
    x = BatchNormalization(momentum=0.95)(x)

    # print(i13.shape)
    x = Concatenate(axis=-1)([x,i13])
    x = Conv3DTranspose(filters=128, kernel_size=(4,4,5), strides=2, padding='same')(x)
    x = Dropout(0.2)(x)
    # print(x.shape)
    x = BatchNormalization(momentum=0.95)(x)
    x = Concatenate(axis=-1)([x,i10])
    x = Conv3DTranspose(filters=64, kernel_size=(6,6,4), strides=2, padding='same')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(momentum=0.95)(x)
    
    x = Reshape(target_shape=[x.shape[1], x.shape[2], x.shape[3]*x.shape[4]])(x)
    
    x = Concatenate(axis=-1)([x,i5])
    x = Conv2DTranspose(filters=64, kernel_size=9, strides=2, padding="same")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Concatenate(axis=-1)([x,i2])
    x = Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding="same")(x)
    xl = Conv2D(filters=1, kernel_size=4, strides=1, padding="same", activation='tanh')(x)

    
    return Model(img, xl)

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.depth_shape = (self.img_rows, self.img_cols, 3)

        # Configure data loader
        self.dataset_name = 'NYU3'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.1, 0.5)

        

        # Build and compile the discriminator
        # self.discriminator = self.build_discriminator()
        # self.discriminator.compile(loss='mse',
        #     optimizer=optimizer,
        #     metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        
        # self.generator.load_weights("/content/drive/My Drive/biglol/lolg.h5")
        self.generator.compile(loss=[berhu_loss],
            optimizer=optimizer,
            metrics=['accuracy', 'mse', rmse])
        self.generator.load_weights("/storage/lolgfull_1d_2.h5")
        '''
        self.generator.trainable = False
        lol = Model(self.generator.input, self.generator.layers[-2].output)
        img_A = Input(shape=self.img_shape)
        bb0 = Conv2D(8, kernel_size=4, strides=1, padding='same', activation='relu')(img_A)
        bb1 = Conv2D(16, kernel_size=4, strides=1, padding='same', activation='relu')(bb0)
        bb2 = Conv2D(32, kernel_size=4, strides=1, padding='same', activation='relu')(bb1)
        bb3 = Conv2D(64, kernel_size=4, strides=1, padding='same', activation='relu')(bb2)
        dd = lol(img_A)
        aa = Concatenate()([dd,bb3])
        aa = Conv2D(64, kernel_size=4, strides=1, padding='same', activation='relu')(aa)
        aa = Concatenate()([aa,bb2])
        aa = Conv2D(32, kernel_size=4, strides=1, padding='same', activation='relu')(aa)
        aa = Concatenate()([aa,bb1])
        aa = Conv2D(16, kernel_size=4, strides=1, padding='same', activation='relu')(aa)
        aa = Concatenate()([aa,bb0])
        oo = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(aa)
        self.generator2 = Model(inputs=img_A, outputs=oo)

        self.generator2.compile(loss=[berhu_loss],
            optimizer=optimizer,
            metrics=['accuracy', 'mse', rmse])
        
        '''



        # # Input images and their conditioning images
        # img_A = Input(shape=self.img_shape)
        # img_B = Input(shape=self.img_shape)

        # # By conditioning on B generate a fake version of A
        # fake_A = self.generator(img_A)

        # # For the combined model we will only train the generator
        # self.discriminator.trainable = False

        # # Discriminators determines validity of translated images / condition pairs
        # valid = self.discriminator([fake_A, img_A])

        # self.combined = Model(inputs=[img_A, img_B], outputs=[fake_A,valid])
        # self.combined.compile(loss=['mse', 'mae'],
        #                       loss_weights=[1, 100],
        #                       optimizer=optimizer)



    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)
        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8, dropout_rate=0.2)
        u2 = deconv2d(u1, d5, self.gf*8, dropout_rate=0.2)
        u3 = deconv2d(u2, d4, self.gf*8, dropout_rate=0.2)
        u4 = deconv2d(u3, d3, self.gf*4, dropout_rate=0.2)
        u5 = deconv2d(u4, d2, self.gf*2, dropout_rate=0.2)
        u6 = deconv2d(u5, d1, self.gf, dropout_rate=0.2)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        # valid = np.ones((batch_size,) )
        # fake = np.zeros((batch_size,) )

        for epoch in range(0,epochs):
            
            
#             gc.collect()
            
            # self.discriminator.save("/content/drive/My Drive/biglol/lold.h5")
            # self.combined.save("/content/drive/My Drive/biglol/lolc.h5")
            # eee = epoch
            try:
                bar = tqdm(total=self.data_loader.n_batches, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                   desc="")
            except:
                bar = tqdm(total=1488, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                   desc="")
            for batch_i, (imgs_B, imgs_A) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                loss2 = self.generator.train_on_batch(imgs_B, imgs_A)
#                 loss = self.generator2.train_on_batch(imgs_B, imgs_A)
                loss = [0,0,0,0,0]
                
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                # print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                #                                                         batch_i, self.data_loader.n_batches,
                #                                                         d_loss[0], 100*d_loss[1],
                #                                                         g_loss[0],
                #                                                         elapsed_time))
                bar.update(1)
                bar.set_description ("[Epoch %d/%d] [Batch %d/%d] "% (epoch, epochs,
                                                                      batch_i, self.data_loader.n_batches)+ "  loss = "+ str(loss2[0]) + ", " + str(loss[0]) + " rmse = " + str(loss2[3]) + ", " + str(loss[3]))
#                 bar.set_description ("[Epoch %d/%d] [Batch %d/%d] "% (epoch, epochs,
#                                                                       batch_i, self.data_loader.n_batches)+ "  loss = "+ str(loss2[0]) + " rmse = " + str(loss2[3]))
                
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    #self.generator2.save("/storage/lolg2full_1d.h5")
                    self.generator.save("/storage/lolgfull_1d_2.h5")

                # If at save interval => save generated image samples
                

    # def sample_images(self, epoch, batch_i):
    #     os.makedirs('/content/drive/My Drive/biglol/images/%s' % self.dataset_name, exist_ok=True)
    #     r, c = 3, 3

    #     imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
    #     # noise = np.random.randn(3, 64, 64, 3)
    #     fake_A = self.generator.predict(imgs_A)

    #     # gen_imgs = np.array([imgs_A, fake_A, imgs_B])
        
    #     imgs_A = 0.5*imgs_A + 0.5
    #     fake_A = 0.5*fake_A + 0.5
    #     imgs_B = 0.5*imgs_B + 0.5
    #     gen_imgs = [imgs_A, fake_A, imgs_B]
    #     # print(imgs_B)
    #     # Rescale images 0 - 1
    #     # gen_imgs = 0.5 * gen_imgs + 0.5

    #     title = ['Condition', 'Generated', 'Original']
    #     os.makedirs('/content/drive/My Drive/biglol/images/%s/%s' % (self.dataset_name, str(epoch)+"_"+str(batch_i)), exist_ok=True)
    #     for i in range(1):
    #         for j in range(3):
    #             path = "/content/drive/My Drive/biglol/images/NYU/"+ str(epoch) + "_" + str(batch_i) +"/" + title[i] +str(j) + ".jpg"
    #             imageio.imsave(path, gen_imgs[i][j])
                
    #     for i in range(1,3):
    #         for j in range(3):
    #             path = "/content/drive/My Drive/biglol/images/NYU/"+ str(epoch) + "_" + str(batch_i) +"/" + title[i] +str(j) + ".png"
    #             plt.imshow(np.reshape(gen_imgs[i][j], (self.img_shape[0],self.img_shape[1])), cmap='Spectral')
    #             plt.savefig(path)
            
    def sample_images(self, epoch, batch_i):
        os.makedirs(folder_path+'images/%s' % self.dataset_name+"3", exist_ok=True)
        r, c = 3, 3

        imgs_B, imgs_A = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)
#         fake_A2 = self.generator2.predict(imgs_B)
        #imgs_A = 0.5*imgs_A +0.5
        imgs_B = 0.5*imgs_B +0.5
        #fake_A = 0.5*fake_A + 0.5
#         fake_A2 = 0.5*fake_A2 + 0.5
        cn = 0
        while cn<3:
            f = fake_A[cn]
            d = imgs_A[cn]
            i = imgs_B[cn]
            f = np.reshape(f, (self.img_rows, self.img_cols))
            d = np.reshape(d, (self.img_rows, self.img_cols))
#            f[:,:,2] = f[:,:,2] *0.5 +0.5
#            d[:,:,2] = d[:,:,2] *0.5 +0.5  
            f = f *0.5 +0.5
            d = d *0.5 +0.5 
            titles = ['Image', 'Gen depth', 'Org depth']
            plt.subplot(3,3,cn*3+1)
            plt.title(titles[0])
            plt.imshow(i, cmap='gray')
            plt.subplot(3,3,cn*3+2)
            plt.title(titles[1])
            plt.imshow(f.astype(np.float32), cmap='gray')
            plt.subplot(3,3,cn*3+3)
            plt.title(titles[2])
            plt.imshow(d, cmap='gray')
            cn+=1

        plt.savefig(folder_path+"images/%s/%d_%d.png" % (self.dataset_name+"3", epoch, batch_i))
        plt.close()
#         np.save(folder_path+"images/%s/%d_%dDEPTH.npy" % (self.dataset_name+"3", epoch, batch_i),fake_A)
#         np.save(folder_path+"images/%s/%d_%dIMGS.npy" % (self.dataset_name+"3", epoch, batch_i),imgs_B)
        cn = 0
#         while cn<3:
#             f = fake_A2[cn]
#             d = imgs_A[cn]
#             i = imgs_B[cn]
#             f = np.reshape(f, (self.img_rows, self.img_cols))
#             d = np.reshape(d, (self.img_rows, self.img_cols))
#             titles = ['Image', 'Gen depth', 'Org depth']
#             plt.subplot(3,3,cn*3+1)
#             plt.title(titles[0])
#             plt.imshow(i, cmap='gray')
#             plt.subplot(3,3,cn*3+2)
#             plt.title(titles[1])
#             plt.imshow(f, cmap='gray')
#             plt.subplot(3,3,cn*3+3)
#             plt.title(titles[2])
#             plt.imshow(d, cmap='gray')
#             cn+=1

#         plt.savefig(folder_path+"images/%s/big%d_%d.png" % (self.dataset_name+"3", epoch, batch_i))
#         plt.close()
      



gan = Pix2Pix()



# gan.combined.load_weights("/content/drive/My Drive/biglol/lolc.h5")
# gan.discriminator.load_weights("/content/drive/My Drive/biglol/lold.h5")
#gan.generator2.load_weights("/storage/lolg2full_1d.h5")
# gan.generator.load_weights("/storage/lolPoint.h5")
# gan.generator.load_weights("/storage/lol3gfull_1d.h5")
gan.generator.summary()
#gan.generator2.summary()
# gan.discriminator.summary()
gan.train(epochs=500, batch_size=32, sample_interval=500)

# for i in range(0,10):
#     gan.sample_images(10000, i)

# a = gan.generator.get_layer(index=-2)
# print(a)

