# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:18:13 2020

@author: richard_dev
"""
import numpy as np
import matplotlib.pyplot as plt

def create_binary_y_variable(y,i):
    y_new=np.copy(y)
    y_new[y==i]=1
    y_new[y!=i]=0
    return y_new

def visualise_number(x):
    x_square=np.reshape(x,[20,20])
    x1=np.linspace(0,19,20)
    x2=np.linspace(0,19,20)
    plt.contourf(x1,x2,x_square,cmap='gray')
    
def sigmoid_function(z):
    g=1/(1+np.exp(-z))
    return g
    
def compute_h(theta,X):
    z=np.dot(X,theta)
    h=sigmoid_function(z)
    return h
    
def compute_J(X,y,theta,m):
    h=compute_h(theta,X)
    J=(1/m)*(-np.dot(y,np.log(h))-np.dot((1-y),(np.log(1-h))))
    return J

def compute_J_with_regularisation(X,y,theta,m,k):
    h=compute_h(theta,X)
    J=(1/m)*(-np.dot(y,np.log(h))-np.dot((1-y),(np.log(1-h))))+(k/(2*m))*\
        np.dot(theta,theta)
    return J

def update_theta(X,y,theta,alpha,m):
    h=compute_h(theta,X)
    theta=theta-(alpha/m)*np.dot(X.transpose(),(h-y))
    return theta

def update_theta_with_regularisation(X,y,theta,alpha,m,k):
    h=compute_h(theta,X)
    #Theta 0
    theta[0]=theta[0]-(alpha/m)*np.dot(X.transpose()[0,:],(h-y))
    #Theta 1 to n
    theta[1:]=theta[1:]-alpha*((1/m)*(np.dot(X.transpose()[1:,:],(h-y)))+\
                              ((k/m)*theta[1:]))
    return theta

def test_classification(x,Theta):
    H=sigmoid_function(np.dot(Theta.transpose(),x))
    number=np.where(H==max(H))
    number=number[0].item()
    return number

def test_classification_all(X,Theta,m):
    z=np.dot(X,Theta)
    H=sigmoid_function(z)
    max_per_row=np.amax(H,1)
    estimated_numbers=np.zeros(m)
    for i in range(m):
        estimated_number=np.where(H[i,:]==max_per_row[i])
        estimated_numbers[i]=estimated_number[0].item()
    return estimated_numbers
    
    