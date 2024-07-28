# -*- coding: utf-8 -*-
__author__ = 'Gerald Schuller'
__copyright__ = 'G.S.'

"""
Simple program to use a convolutional neural network to obtain a matched filter, or an analysis filter bank, with filtering followed by downsampling.
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

def generate_dummy_data():
    #Method to generate some artificial data in an numpy array form in order to fit the network.
    #:return: X, Y numpy arrays used for training, X: Input, Y: Target
    
    #Input signal X, a Ramp function:
    X= np.hstack((np.zeros((1,9)),np.expand_dims(np.arange(5),axis=0),np.zeros((1,9)))) #ramp as simple signal to detect
    #Make it unit L2 norm:
    X= X/np.sqrt(np.dot(X,X.transpose()))
    #Make input a shape that keras expects,
    #input X with shape (batch, length, channels), channels: e.g. RGB
    #https://stackoverflow.com/questions/43235531/convolutional-neural-network-conv1d-input-shape
    X=X.transpose()  #signal in the middle dimension
    X = np.expand_dims(X, axis=0)  #add batch dimension (here only 1 batch)
    
    #Target function Y, the desired output:
    Y = np.zeros((1,23))
    Y[0,16]=1 #Detecting the signal at its end (for convolution padding='causal').
    #Make target a shape keras expects, same as input X shape:
    Y=Y.transpose()
    Y=np.expand_dims(Y, axis=0)
    return X, Y


def generate_model():
    #    Method to construct a convolutional neural network using keras and theano.
    #    :return: Trainable object
    
    # Define the model. 
    model = Sequential()
    #Obtain a linear filter from a convolutional layer, similar to a matched filter:
    model.add(Conv1D(filters=1, kernel_size=(8), strides=1, padding='causal', activation="linear", use_bias=False, kernel_initializer='glorot_uniform', input_shape=(23,1)) )
    #uniform initialization:
    #model.add(Conv1D(filters=1, kernel_size=(8), strides=1, padding='causal, activation="linear", use_bias=False, kernel_initializer='uniform', input_shape=(23,1)) )
    
    # Compile appropriate theano functions
    #losses: https://keras.io/losses/
    #mean_squared_error ('mse'), mean_absolute_error(y_true, y_pred), mean_squared_logarithmic_error,...
    #model.compile(loss='mean_squared_error', optimizer='sgd')
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

if __name__ == '__main__':
    #   Demonstration on using the code.
    X, Y = generate_dummy_data() # Acquire Training Dataset
    print("Input X[0,:,0]=", X[0,:,0], "X.shape=", X.shape )
    print("Target Y[0,:,0]=", Y[0,:,0], "Y.shape=", Y.shape)

    model = generate_model()     # Compile an neural net
    print("Train model:")
    model.fit(X, Y, epochs=5000, batch_size=1)  #use 5000 epochs or iterations for training
    model.evaluate(X, Y,  batch_size=1, verbose=1)    
    predictions=model.predict(X) # Make Predictions based on the obtained weights
    ww = model.get_weights()   #read obtained weights
    #weight format for Conv1d:
    #[0: filter weights, 1: bias for first layer]
    #[filterlength, channels (subbands) in previous layer, neurons/filters in this layer] 
    weights=ww[0][:,0,0]
    
    print("Predictions[0,:,0]= ", predictions[0,:,0])
    print("weights= ", weights)

    #model.save_weights('weights.hdf5') #save weights to file
    with open("convnetweights.pickle", 'wb') as weightfile:
       pickle.dump(ww, weightfile)
    
    plt.plot(predictions[0,:,0])
    plt.title('The Conv. Neural Network Output')
    plt.figure()
    plt.plot(weights)
    plt.title('The Weights')
    plt.figure()
    plt.plot(X[0,:,0])
    plt.title('The Input Signal')
    plt.show()
    

