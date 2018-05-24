#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:17:59 2018

@author: mustafa
"""
#Linear regression with one variable and vectorization
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
def plot(theta,filename):#theta is matris
    x,y=get_attribütes_from_file(filename)
    a=np.arange(150)
    plt.scatter(x,y)
    plt.scatter(a,theta.item(0)*a+theta.item(1))
    plt.show()
    
def get_attribütes_from_file(filename):#get attribütes using pandas
    file=pd.read_csv(filename)
    x=file[file.columns.values[0]]
    y=file[file.columns.values[1]]
    return x,y
def gradientDescentWithvectorization(x,y):#calculate thetas using numpy
    x=np.vstack([x,np.ones(99)])
    theta=np.matrix('1,1')
    for i in range(1000):
        theta=theta-1/(len(y))*0.0001*(theta*x-np.matrix(y))*x.T#learning rate is 0.0001
    print("ERROR: ",getErrorWithVectorization(theta,x,len(y),y))
    return theta
def getErrorWithVectorization(theta,x,m,y):# get error using numpy 
    return 1/(2*m)*np.sum(*np.square(theta*x-np.matrix(y)))
x,y=get_attribütes_from_file("linear.csv")
theta=gradientDescentWithvectorization(x,y)
plot(theta,"linear.csv")
