# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:02:20 2018

@author: Neil sharma
"""

from keras.preprocessing.image import img_to_array,load_img
#from sklearn.preprocessing import StandardScaler
import numpy as np 
import os 
from scipy.misc import imresize 
#import argparse

image_store = []
    
    

#List of all Videos in the Source Directory. 
#videos=os.listdir(video_source_path)


#Make a temp dir to store all the frames
#os.mkdir(video_source_path + '/frames')
ped1_path = r"ucsd\UCSDped1\Train"
paths = os.listdir(ped1_path)

for path in paths:
    framepath = ped1_path + "/" + path
    
    """for video in videos:
        os.system( 'ffmpeg -i {}/{} -r 1/{} {}/frames/%03d.jpg'.format(video_source_path,video,fps,video_source_path))"""
    images = os.listdir(framepath)
    for image in images:
            #image_path = framepath + "/" + image
            image_path = framepath + "/" + image
            img = load_img(image_path)
            img = img_to_array(img)
                
                
            #Resize the Image to (224,224,3) for the network to be able to process it. 
            img = imresize(img,(224,224,3))
                
            #Convert the Image to Grayscale
                
                
            g = 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]
                
            image_store.append(g)
            #store(image_path)


image_store = np.array(image_store)
image_store.shape
a, b, c = image_store.shape
        #Reshape to (227,227,batch_size)
image_store.resize(b,c,a)
#Normalize
image_store=(image_store-image_store.mean())/(image_store.std())
#Clip negative Values
image_store=np.clip(image_store,0,1)
np.save('training.npy',image_store)
#Remove Buffer Directory
#os.system('rm -r {}'.format(framepath))
