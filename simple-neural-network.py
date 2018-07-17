# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 19:33:39 2018

@author: mustafa
"""
import numpy as np
from sklearn import datasets as ds
from sklearn.metrics import accuracy_score # to estimate accuracy



class NeuralNetwork(object):
    def __init__(self):
          self.inputSize = 4
          self.outputSize = 3
          self.hiddenSize = 5
         #random initiliaze weights and bias
          self.W1 = np.random.randn(self.inputSize, self.hiddenSize)*0.01
          self.W2 = np.random.randn(self.hiddenSize, self.outputSize)*0.01
          self.b2=np.random.randn(1, self.outputSize)*0.01
          self.b1=np.random.randn(1,self.hiddenSize)*0.01
         
    def forward(self,x):# forward propogation
        self.z1=np.dot(x,self.W1)+self.b1
        self.a1=self.sigmoid(self.z1)
        self.z2=np.dot(self.a1,self.W2)+self.b2
        o=self.sigmoid(self.z2)
        return o
        
        
    def sigmoid(self, s,derivate=False):
      if derivate:# if want to calculate derivate
          return s * (1 - s)
          
      return 1/(1+np.exp(-s))
   
    def backward(self,x,y,o,regularize,lambd=0.008,learning_rate=0.1325):
        
        
        #calculate derivates
        self.dz2=np.multiply((y-o),self.sigmoid(self.z2,derivate=True))
        
        self.dw2=(1/150*np.dot(self.a1.T,(self.dz2)))
        
        self.db2=1/150*np.sum(self.dz2,axis=0,keepdims=True)
        
        self.dz1=1/150*np.multiply(self.dz2.dot(self.W2.T),self.sigmoid(self.z1,derivate=True))
        
        self.dw1=1/150*np.dot(x.T,(self.dz1))
        
        self.db1=1/150*np.sum(self.dz1,axis=0,keepdims=True)
        if regularize:   #if want to regularize
            self.dw2+=lambd*self.W2
            self.dw1+=+lambd*self.W1
            
        #update weights and bias
        self.W1=self.W1-learning_rate*self.dw1
        
        self.W2=self.W2-learning_rate*self.dw2
        
        self.b1=self.b1-learning_rate*self.db1
        
        self.b2=self.b2-learning_rate*self.db2
        
        
    def train(self,x,y,regularize=False,epoch=1000):
        for i in range(epoch):
             o=self.forward(x)
             print ("epoch: ",i,"\n","Loss: " , NN.calculate_loss(y,o)) 
             self.backward(x,y,o,regularize)
       
    
    def accuracy(self,y,o): #calculate accuracy
        return accuracy_score(y,o.argmax(axis=1))
    
    
    def calculate_loss(self,y,o):# calculate loss
        return np.mean(np.square(y - NN.forward(x)))









