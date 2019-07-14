# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:57:40 2018

@author: Neil sharma
"""


import numpy as np
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
#from keras.optimizers import adam
from sklearn import svm

"""def load_model():
    '''
	Return the model used for abnormal event 
    '''

	dbn = DBN([X_train.shape[1:2], [300,300], 1], learn_rates = 0.3)

	return dbn
"""


#load_model()
#n_epochs = 50

X_train = np.load('training.npy')

X_train = X_train.reshape(-1,224,224,1)
Y_train = X_train.copy()
#a = X_train.shape
#np.max(X_train)
X_test = np.load('testing.npy')
X_test = X_test.reshape(-1,224,224,1)
Y_test = X_test.copy()
x, y = 224, 224
inChannel = 1

input_img = Input(shape = (x, y, inChannel))

epochs = 1
batch_size = 10

#validation_data = (X_test, Y_test)

#Encoder:
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
encoded = MaxPooling2D(pool_size = (2, 2))(conv3)


#Decoder:
conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
up1 = UpSampling2D((2,2))(conv4)
conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2,2))(conv5)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)


autoencoder = Model(input_img, decoded)
autoencoder.compile(loss = 'mean_squared_error', optimizer = 'adam')
autoencoder.summary()
autoencoder.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose = 1)
autoencoder.save("autoencoder_dbn.h5")

#encoded.shape
#ocsvm = autoencoder.predict(X_train)

#ocsvm = np.array(ocsvm)
#ocsvm.shape
#Normalize
#ocsvm = (ocsvm - ocsvm.mean()) / (ocsvm.std())
#np.save("svm.npy", ocsvm)


encoder = Model(input_img, encoded)

ocsvm_train = encoder.predict(X_train)
ocsvm_train.shape

X_train = ocsvm_train
dataset_size = len(X_train)
X_train = X_train.reshape(dataset_size, -1)



ocsvm_test = encoder.predict(X_test)
ocsvm_test.shape

X_test = ocsvm_test
dataset_size = len(X_test)
X_test = X_test.reshape(dataset_size, -1)

#X_train.reshape(1, -1)
#Y_train = X_train.copy()
#a = X_train.shape


'''
#One Class SVM
classifier = svm.OneClassSVM(nu = 0.001, kernel = "rbf", gamma = 0.01)
classifier.fit(training_feature_vector_list)
'''


#epochs = 50
#batch_size = 1


#One Class SVM
classifier = svm.OneClassSVM(nu = 0.03761600681140911, kernel = "rbf", gamma = 0.00005)
classifier.fit(X_train)

#Prediction
ans = classifier.predict(X_test)
