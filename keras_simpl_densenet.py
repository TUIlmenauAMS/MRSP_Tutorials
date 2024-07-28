# -*- coding: utf-8 -*-
__author__ = 'Gerald Schuller'
__copyright__ = 'G.S.'

"""
Simple program to implement a convolutional neural network using a dense network, for real time audio implementations.
Gerald Schuller, July 2017.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv1D
from keras.constraints import unit_norm
import numpy as np
import matplotlib.pyplot as plt
import sys

if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle

from keras_simpl_convnet import generate_dummy_data

if __name__ == '__main__':
   
   #Open weights file:
    with open("convnetweights.pickle", 'rb') as weightfile:
       weights=pickle.load(weightfile)
    print("weights[0].shape=", weights[0].shape)
    
    #Converting weights from Convolutional layer to Dense layer:
    weights[0]=weights[0][:,:,0] #remove channel dimension for the weights
    print("weights[0].shape=", weights[0].shape)
    #Flipping the impulse response (filter) dimension from the convolutional layer
    #(Convolution is correlation with flipped filter weights):
    #weights[0]=np.flip(weights[0], axis=0)
    filtlen=len(weights[0])
    print("filtlen=", filtlen)
    
    #Setup dense neural network model:   
    model = Sequential()
    model.add(Dense(units=1, activation='linear', use_bias=False, input_shape=(filtlen,)) )
    model.set_weights(weights)
    
    X, Y = generate_dummy_data() # Acquire signal
    siglen=len(X[0,:,0])
    prediction=np.zeros(siglen-filtlen+1)
    #Transpose the signal for the dense layer, to obtain a row vector to be multiplied from the left 
    #to the weight vector in Dense:
    X=np.transpose(X,axes=(0,2,1))
    
    #Loop over the entire signal in single steps until the end, in each step try to decode:
    for n in range(siglen-filtlen):
       #cut out the current signal block:
       Xblock=X[0,:,n:n+filtlen]
       #print("Xblock.shape=", Xblock.shape)
       
       #Estimate the likelyhood of each character for the current block:
       prediction[n]=model.predict(Xblock)
       
    plt.plot(prediction)
    plt.title('The Conv. Neural Network Output')
    plt.show()
    
