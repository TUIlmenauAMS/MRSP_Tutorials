# -*- coding: utf-8 -*-
__author__ = 'Gerald Schuller'
__copyright__ = 'G.S.'

"""
Simple program to implement a convolutional neural network, or an analysis filter bank, using a keras dense net, for real time audio or wireless processing. For a keras convolutional network it is assumed that the entire data is already in memory, but in real time processing it needs to be processed as it arrives, sample- or block-wise.
Instead of Convent "sliding" the filter along the samples, here we slide the samples along the filter, implemented using a Dense net.
Gerald Schuller, May 2018.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
import matplotlib.pyplot as plt
import sys

if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle
   
filtlen=8
   
def generate_model():
    #    Method to construct a convolutional neural network using keras and theano.
    #    :return: Trainable object
    
    # Define the model. 
    model = Sequential()
    #Obtain a linear filter from a convolutional layer, similar to a matched filter:
    model.add(Dense(units=1, activation='linear', use_bias=False, input_shape=(filtlen,)))
    
    return model
    
def generate_data():
    #Method to generate some artificial data in an numpy array form in order to fit the Dense network.
    #:return: data X: I
    
    #Input signal X, a Ramp function:
    X= np.hstack((np.zeros((1,9)),np.expand_dims(np.arange(5),axis=0),np.zeros((1,9)))) #ramp as simple signal to detect
    #Make it unit L2 norm:
    X= X/np.sqrt(np.dot(X,X.transpose()))
    #Make input a shape that keras expects,
    #For Dense net: input X with shape (batch, length), 
    
    return X

    
if __name__ == '__main__':
    #   Demonstration on using the code.
    
    #Open weights file:
    with open("convnetweights.pickle", 'rb') as weightfile:
       weights=pickle.load(weightfile)
    print("weights[0].shape=", weights[0].shape)
    
    #Convert Conv1D to Dense weights:################################
    #weight format for Conv1d:
    #[0: filter weights, 1: bias for first layer]
    #[filterlength, channels (subbands) in previous layer, neurons/filters in this layer]  
    #weight format for Dense:
    #[0: weights, 1: bias for first layer]
    #[total weights=filterweights * channels (subbands or neurons) in previous layer (time-reversed), neurons/filters in this layer] 
    #remove "channels" dimension, since we only have 1 channel here:
    weights[0]=weights[0][:,0,:]
    print("weights[0].shape=", weights[0].shape)
    #flip filter or weights dimension, because signal will be flipped. For Keras version > 2.4.3.
    weights[0]=np.flip(weights[0], axis=0)
    
    model = generate_model() 
    #assign the converted weights:
    model.set_weights(weights)
    
    #Aquire Input signal X, the Ramp function:
    X = generate_data() 
    print("X.shape=", X.shape)
    #initialize "sliding" block buffer for the signal, as input for the Dense net:
    signalblock=np.zeros((1,filtlen))
    
    predictionsbuffer=np.zeros(X.shape[1])
    #Simulate the samples coming samplewise from e.g. a sound card:
    print("X.shape[1]=",X.shape[1])
    for sampind in range(X.shape[1]):
       #"slide" buffer contest one sample "up":
       signalblock[:,1:]=signalblock[:,0:-1]
       #assign current value to "buttom". 
       #This flips the signal, since the newest value appears at the lowest index:
       signalblock[0,0]= X[0,sampind]
       #obtain model output for the block input:
       prediction=model.predict(signalblock)
       print("prediction.shape=", prediction.shape)
       #Outputs are collected in the buffer:
       predictionsbuffer[sampind]=prediction
       
    print("predictionsbuffer.shape=", predictionsbuffer.shape)  
    plt.plot(predictionsbuffer)
    plt.title('The Conv. Neural Network Output')
    plt.figure()
    plt.plot(weights[0])
    plt.title('The Weights')
    plt.figure()
    plt.plot(X[0,:])
    plt.title('The Input Signal')
    plt.show()
    
