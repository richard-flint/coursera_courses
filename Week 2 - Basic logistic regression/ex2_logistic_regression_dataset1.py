# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:50:10 2020

@author: richa
"""
#Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import statistics as stat
import math

# Load data
data = pd.read_csv('ex2data1.txt',sep=",", header=None,names=["x1","x2","y"])

#Define variables
X=data[['x1','x2']].values
y=data['y'].values
x1=data['x1'].values
x2=data['x2'].values

#Add x0 to data
m=data.shape[0]
x0=np.ones(m)
X=np.vstack((x0,x1,x2))
X=X.transpose()

#Feature scaling
x0_n=x0
x1_n=(x1-stat.mean(x1))/(max(x1)-min(x1))
x2_n=(x2-stat.mean(x2))/(max(x2)-min(x2))
X_transpose_n=np.array([x0_n,x1_n,x2_n])
X_n=X_transpose_n.transpose()

#Plot data
plt.figure(1)
x1_y0=x1[y == 0]
x2_y0=x2[y == 0]
x1_y1=x1[y != 0]
x2_y1=x2[y != 0]
plt.figure(1)
plt.plot(x1_y0,x2_y0,'o')
plt.plot(x1_y1,x2_y1,'x')

#Plot feature scaled data
plt.figure(2)
x1n_y0=x1_n[y == 0]
x2n_y0=x2_n[y == 0]
x1n_y1=x1_n[y != 0]
x2n_y1=x2_n[y != 0]
plt.plot(x1n_y0,x2n_y0,'o')
plt.plot(x1n_y1,x2n_y1,'x')

#Initialise theta
theta=np.array([0,0,0],dtype=np.float64)

#****************************#
#*** Define key functions ***#
#****************************#

#Function for computing hypothesis
def compute_h(X,theta):
    z=np.dot(theta,X.transpose())
    h=sigmoid(z)
    return h

#Sigmoid function
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

#Compute cost
def compute_cost(X,y,theta,m):
    h=compute_h(X,theta)
    J=(-1/m)*(np.dot(y,np.log(h))+np.dot((1-y),np.log(1-h)))
    return J

#****************************#
#****************************#

#Initialise key variables
alpha=30
number_of_iterations=50000
J_history=np.zeros(number_of_iterations+1)
theta_history=np.zeros((number_of_iterations+1,3))
theta=np.array([0,0,0],dtype=np.float64)

#Calculate initial cost
J_history[0]=compute_cost(X_n,y,theta,m)

#Implement gradient descent
for i in range(number_of_iterations):
    
    #Compute the hypothesis
    h=compute_h(X_n,theta)
    
    #Calculate new theta
    theta[0]=theta[0]-(alpha/m)*np.dot((h-y),x0_n)
    theta[1]=theta[1]-(alpha/m)*np.dot((h-y),x1_n)
    theta[2]=theta[2]-(alpha/m)*np.dot((h-y),x2_n)
    
    #Save values of J to graph
    J_history[i+1]=compute_cost(X_n,y,theta,m)
    theta_history[[i+1],[0]]=theta[0]
    theta_history[[i+1],[1]]=theta[1]
    theta_history[[i+1],[2]]=theta[2]

#Plot J
plt.figure(3)
plot_x=np.linspace(0, number_of_iterations,number_of_iterations+1)
plot_y=J_history
plt.plot(plot_x,plot_y)

#Plot the decision boundary
x0_plot=np.ones(101)
x1_plot=np.linspace(-0.5,0.5,101)
decision_boundary=(-theta[0]*x0_plot-(theta[1]*x1_plot))/theta[2]
plt.figure(7)
plt.plot(x1_plot,decision_boundary)
plt.plot(x1n_y0,x2n_y0,'o')
plt.plot(x1n_y1,x2n_y1,'x')
print("Theta = ",theta)





