# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:59:59 2018

@author: Neil sharma
"""


import numpy as np 
from sklearn import svm
from keras.models import load_model
#from keras.models import Model
from keras.layers import Input


x, y = 224, 224
inChannel = 1

input_img = Input(shape = (x, y, inChannel))
"""def load_model():
    '''
	Return the model used for abnormal event 
    '''

	dbn = DBN([X_train.shape[1:2], [300,300], 1], learn_rates = 0.3)

	return dbn
"""
model = load_model('autoencoder_dbn.h5')
#load_model()
#n_epochs = 50
encoded = model.encoded
encoder = model(input_img, model.encoded)

X_train = np.load('svm.npy')

'''
frames = X_train.shape[2]

#Need to make number of frames divisible by 10
frames = frames-frames%10

X_train = X_train[:,:,:frames]
X_train = X_train.reshape(-1,224,224,1)'''

Y_train = X_train.copy()
#a = X_train.shape


epochs = 50
batch_size = 1


#One Class SVM
classifier = svm.OneClassSVM(nu = 0.001, kernel = "rbf", gamma = 0.01)
classifier.fit(X_train)


classifier.save('ocsvm_dbn.h5')