''' Source Code for the SpatioTemporal AutoEncoder as described in the paper

Abnormal Event Detection in Videos using Spatiotemporal Autoencoder

Implemented in keras

The model has over a Million trainable Params so I recommend training it on a GPU.


The model takes input a batch of 10 of Video frames of size (224,224) (grayscaled)

Extracts spatial and temporal Information and computes the reconstruction loss by Euclidean Distance b/w
original batch and Reconstructed batch

'''

from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn import svm


def load_model():
    
    x, y = 224, 224
    inChannel = 1
    
    input_img = Input(shape = (x, y, inChannel))
    
	#Encoder:
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
            
    
    #Decoder:
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    up1 = UpSampling2D((2,2))(conv4)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2,2))(conv5)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
    autoencoder.summary()
    
    
    #One Class SVM
    classifier = svm.OneClassSVM(nu = 0.001, kernel = "rbf", gamma = 0.01)

    
    
    return autoencoder



