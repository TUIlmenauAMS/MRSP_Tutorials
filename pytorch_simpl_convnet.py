# -*- coding: utf-8 -*-
__author__ = 'Gerald Schuller'
__copyright__ = 'G.S.'

"""
Simple program to use a convolutional neural network to obtain a matched filter, or an analysis filter bank, with filtering followed by downsampling, using pytorch.
Gerald Schuller, July 2018.
"""

import torch 
import torch.nn as nn

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
    #input X with shape (batch, channels, length), channels: e.g. RGB
    #https://pytorch.org/docs/stable/nn.html
    X = np.expand_dims(X, axis=0)  #add batch dimension (here only 1 batch)
    
    #Target function Y, the desired output:
    Y = np.zeros((1,30))
    Y[0,16]=1 #Detecting the signal at its end (for convolution padding='causal').
    #Make target a shape pytorch expects, same as input X shape:
    Y=np.expand_dims(Y, axis=0)
    return X, Y

if __name__ == '__main__':
    #   Demonstration on using the code.
    X, Y = generate_dummy_data() # Acquire Training Dataset
    #print("Input X[0,:,0]=", X[0,:,0], "X.shape=", X.shape )
    #print("Target Y[0,:,0]=", Y[0,:,0], "Y.shape=", Y.shape)
    X=torch.from_numpy(X)
    Y=torch.from_numpy(Y)
    X=X.type(torch.Tensor)
    Y=Y.type(torch.Tensor)
    #X = torch.randn(1, 1, 23)
    #Y = torch.randn(1, 1, 30)
    #print("Input X[0,:,0]=", X[0,:,0], "X.shape=", X.shape )
    #print("Target Y[0,:,0]=", Y[0,:,0], "Y.shape=", Y.shape)
    
    print("Generate Model:")
    #model = generate_model()     # Compile an neural net
    #input size (N,Cin,L), N is a batch size, C denotes a number of channels, L is a length of signal sequence.
    #padding=kernel_size-1 corresponds to "causal" in Keras:
    model= nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=8, stride=1, padding=7,groups=1, bias=False),)
    #model= nn.Conv1d(1,1,kernel_size=8)
    print("Def. loss function:")
    loss_fn = nn.MSELoss()
    #learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters())#, lr=learning_rate)
    
    for epoch in range(5000):
       Ypred=model(X)
       loss=loss_fn(Ypred, Y)
       if epoch%100==0:
          print(epoch, loss.item())
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
    
    Ypred=model(X)
    Ypred=Ypred.data.numpy()
    print("Predictions= ", Ypred[0,0,:])
    weights=list(model.parameters())
    print("Pytorch weights=", weights[0].data.numpy())
    #Save in Pytorch format:
    torch.save(model.state_dict(), 'model.ckpt')
    #save weights to Pickle file:
    with open("pytorchconvnetweights.pickle", 'wb') as weightfile:
       pickle.dump(weights, weightfile)
    
    plt.plot(Ypred[0,0,:])
    plt.title('The Conv. Neural Network Output')
    plt.figure()
    plt.plot(weights[0].data.numpy()[0,0,:])
    plt.title('The Pytorch Weights, corr. instead of conv is used!')
    plt.figure()
    plt.plot(X[0,0,:].numpy())
    plt.title('The Input Signal')
    plt.show()
    
