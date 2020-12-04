# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:58:13 2020

@author: 91948
"""
from tkinter import *
import os
import tkinter.messagebox
import tkinter.filedialog
import math
import numpy as np
import cv2
from skimage.transform import resize
from PIL import ImageTk, Image
from worker import work, draw, readFolder
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as backend
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images

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

custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
#model = load_model("nyu.h5",  custom_objects=custom_objects, compile=False 'rmse':rmse})
model = load_model('nyu.h5', custom_objects=custom_objects, compile=False)
class ImageWindow:
    def __init__(self, master):

        #defining master window
        self.master=master
        master.geometry("1000x800")
        master.title("2D -> 3D")

        #defining variables
        self.images=[] #rgb
        self.depths=[] #depth
        self.current_image=None
        
        #defining widgets for browsing window
        self.background_image=ImageTk.PhotoImage(file="head.png",format="png",master=self.master)
        self.background_canvas=Canvas(master, height=800, width=1000, bg="cornsilk2")
        self.image=self.background_canvas.create_image(540,200,image=self.background_image, anchor=CENTER, tags="background")
        
        # gay stuff
        self.location_entry=Entry(self.background_canvas, justify=CENTER, width=50, text="Enter image path here.")
        self.browse_button=Button(self.background_canvas, text="Browse", activebackground="cornsilk3", bg="cornsilk2", command=self.browsefunc,state=NORMAL)
        # next button
        self.analyse_image_button=Button(self.background_canvas, text="Analyse Image", activebackground="cornsilk3", bg="cornsilk2", command=self.next_buttons, state=NORMAL)
        # welcometext
        #self.welctext=Label(self.background_canvas, fg="brown", bg="cornsilk2", font=("Courier", 40), text="lol is lol" )
        
        #placing widgets for resuly window
        self.image_canvas=Canvas(self.background_canvas, height=400, width=500, bg="cornsilk2")
        self.next_button=Button(self.background_canvas, text="Next", activebackground="cornsilk3", bg="cornsilk2", command=self.get_next_image)
        self.next_button2=Button(self.background_canvas, text="Next", activebackground="cornsilk3", bg="cornsilk2", command=self.gen_point_cloud)
        self.next_button3=Button(self.background_canvas, text="Next", activebackground="cornsilk3", bg="cornsilk2", command=self.get_next_image)
        self.new_image=Button(self.background_canvas, text="New Image", command=self.prev_buttons)
        self.close_button=Button(self.background_canvas, text="Close", width=10, activebackground="cornsilk3", bg="cornsilk2", command=self.close_window)
        self.prediction_label=Message(self.background_canvas, text="Character: ", bg="cornsilk2")
        self.retrain_button=Button(self.background_canvas, text="Retrain Model", activebackground="cornsilk3", bg="cornsilk2", command=lambda:train(True))
        
        #placing widgets for 
        self.background_canvas.pack()
        self.location_entry.place(relx=0.3, rely=0.4)
        self.browse_button.place(relx=0.5, rely=0.43)
        self.analyse_image_button.place(relx=0.8, rely=0.9)
        
        #self.welctext.place(relx=0.2, rely=0.1)
        
    def prev_buttons(self):
        self.background_canvas.delete(self.image)
        self.next_button2.place_forget()
        self.new_image.place_forget()
        self.close_button.place_forget()
        self.prediction_label.place_forget()
        self.analyse_image_button.config(state=ACTIVE)
        self.browse_button.config(state=ACTIVE)
        self.background_image=ImageTk.PhotoImage(file="head.png",format="png",master=self.master)
        self.image=self.background_canvas.create_image(540,200,image=self.background_image, anchor=CENTER, tags="background")
        self.background_canvas.pack()
        self.location_entry.place(relx=0.3, rely=0.4)
        self.browse_button.place(relx=0.5, rely=0.43)
        self.analyse_image_button.place(relx=0.8, rely=0.9)
    
    def rerun(self):
        pass
    
    def close_window(self):
        self.master.destroy
        exit(0)
    
    def gen_point_cloud(self):
        self.background_canvas.delete(self.image)
        self.next_button2.place_forget()
        self.new_image.place_forget()
        self.close_button.place_forget()
        self.prediction_label.place_forget()
        draw(self.points)
        self.analyse_image_button.config(state=ACTIVE)
        self.browse_button.config(state=ACTIVE)
        self.background_image=ImageTk.PhotoImage(file="head.png",format="png",master=self.master)
        self.image=self.background_canvas.create_image(540,200,image=self.background_image, anchor=CENTER, tags="background")
        
        # gay stuff

        # welcometext
        #self.welctext=Label(self.background_canvas, fg="brown", bg="cornsilk2", font=("Courier", 40), text="lol is lol" )
        
        #placing widgets for resuly window
        
        #placing widgets for 
        self.background_canvas.pack()
        self.location_entry.place(relx=0.3, rely=0.4)
        self.browse_button.place(relx=0.5, rely=0.43)
        self.analyse_image_button.place(relx=0.8, rely=0.9)
    
    def browsefunc(self):
        self.browser = tkinter.filedialog.askdirectory(initialdir = "/",title = "Select folder")
        self.location_entry.delete(0,END)
        self.location_entry.insert(0,self.browser)
        #print(self.browser)

    def next_buttons(self):
        path = self.location_entry.get()
        if not os.path.isdir(path):
            tkinter.messagebox.showinfo("Invalid directory.","Please enter a valid file path.")
            return
        self.background_canvas.delete(self.image)

        self.analyse_image_button.config(state=DISABLED)
        self.browse_button.config(state=DISABLED)
        #self.model_output=predict(self.location_entry.get())
        '''
        self.i=0
        print(self.model_output[-1])
        self.images=[x[0] for x in self.model_output[:-1]]
        self.predictions=[self.clean_predictions(x[1]) for x in self.model_output[:-1]]
        self.final_predictions=self.model_output[-1]
        
        '''
        # delete other buttons
        self.location_entry.place_forget()
        self.browse_button.place_forget()
        self.analyse_image_button.place_forget()
        #self.welctext.place_forget()
        
        self.points = readFolder(path, model)
        
        #placing widgets
        self.background_canvas.pack()
        '''
        self.image_canvas.place(anchor=CENTER, relx=0.5, rely=0.47)
        img = ImageTk.PhotoImage(file="plot.png",format="png",master=self.image_canvas)
        self.image_canvas.create_image(0, 0, anchor=CENTER, image=img)
        '''
        self.background_image=ImageTk.PhotoImage(file="plot.png",format="png",master=self.master)
        self.image=self.background_canvas.create_image(500,440,image=self.background_image, anchor=CENTER, tags="background")

        #self.change_image(self.images[0])
        self.set_label_texts()
        #self.retrain_button.place(anchor=CENTER, relx=0.9, rely=0.1)
        self.next_button2.place(anchor=CENTER, relx=0.95, rely=0.03)
        self.new_image.place(anchor=CENTER, relx=0.09, rely=0.03)
        self.close_button.place(anchor=CENTER, relx=0.9, rely=0.9)
        self.prediction_label.place(anchor=CENTER, relx=0.5, rely=0.85)

    def change_image(self, image):
        #function to set the image canvas to another image in the form of a numpy array 
        pil_image=Image.fromarray(image.astype("uint8"))
        scale = max(pil_image.size[0]/500, pil_image.size[1]/400)
        self.current_image = ImageTk.PhotoImage(image=pil_image.resize(tuple([int(x/scale) for x in pil_image.size])))
        self.image_canvas.create_image((250,250),image=self.current_image,anchor=CENTER)

    def set_label_texts(self):
        #sets label texts based on what labels are applicable
        self.prediction_label.config(text="Depth text goes here")

    def get_next_image(self):
        #self.depth=ImageTk.PhotoImage(file="bg.jfif",format="gif -index 2",master=root)
        #self.background_canvas=Canvas(self.master, height=800, width=1000, bg="cornsilk2")
        #self.image=self.background_canvas.create_image(250,400,image=self.depth, anchor=CENTER, tags="depth")
        '''
        self.i=(self.i+1)%(3+len(self.images))
        if self.i-len(self.images) in range(3):
            self.image_canvas.delete("all")
            predict_texts=[open("D:\\ML\\Scenes\\"+x).read() for x in self.final_predictions]
            colors=["green", "orange", "red"]
            self.image_canvas.create_text(250,200, text=predict_texts[self.i-len(self.images)], fill=colors[self.i-len(self.images)], width=500, anchor=CENTER)
        else:
            self.image_canvas.delete("all")
            self.change_image(self.images[self.i])
            self.set_label_texts()
        '''
        
    
    def clean_predictions(self, predict_array):
        s=[]
        for entry in predict_array:
            curr=""
            for items in entry:
                #print(entry,"\n")
                if type(items)==type("S"):
                    curr+=items+": "
                else:
                    curr+=("Maybe " if items[1] else "")+items[0]
            s+=[curr]
        #print("here:",s)
        return "\n".join(s)

root=Tk()
root.resizable(False, False)
gui=ImageWindow(root)
root.mainloop()
