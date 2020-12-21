# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:59:44 2020

@author: richard_dev
"""

import numpy as np

def compute_cost_reg(X,y,m,k,theta):
    h=compute_h(theta,X)
    J=(1/(2*m))*np.dot(h-y,h-y)+(k/(2*m))*np.dot(theta,theta)
    return J

def compute_cost_unreg(X,y,m,theta):
    h=compute_h(theta,X)
    J=(1/(2*m))*np.dot(h-y,h-y)
    return J
    
def compute_h(theta,X):
    h=np.dot(theta,X.T)
    return h

def compute_grad_unreg(X,y,theta,m):
    h=compute_h(theta,X)
    D=(1/m)*np.dot(h-y,X)
    return D

def compute_grad_reg(X,y,theta,m,k):
    h=compute_h(theta,X)
    D=(1/m)*np.dot(h-y,X)+(k/m)*theta
    return D

def lin_reg(X,y,m,k,alpha,number_of_iterations,n):
    
    #Initialise theta
    theta=np.ones(n+1,dtype=np.float64)
    
    #Initialise history variables
    J_history=np.zeros(number_of_iterations+1,dtype=np.float64)
    theta_history=np.zeros([number_of_iterations+1,n+1])
    
    #Compute and save initial cost
    J=compute_cost_reg(X,y,m,k,theta)
    J_history[0]=J
    
    #Save initial theta
    theta_history[0,:]=theta

    #Run gradient descent
    for ind in range(number_of_iterations):

        #Compute gradients
        D_unreg=compute_grad_unreg(X,y,theta,m)
        D_reg=compute_grad_reg(X,y,theta,m,k)
        
        #Update theta
        theta[0]=theta[0]-alpha*D_unreg[0]
        theta[1:]=theta[1:]-alpha*D_reg[1:]
        
        #Save theta and J
        J=compute_cost_reg(X,y,m,k,theta)
        J_history[ind+1]=J
        theta_history[ind+1,:]=theta
        
    #Return values
    return theta,J,J_history,theta_history

def create_Xpoly(X,N,m):
    Xpoly=np.zeros([m,N+1])
    Xpoly[:,0]=np.copy(X[:,0])
    Xpoly[:,1]=np.copy(X[:,1])
    for ind in range(2,N+1):
        Xpoly[:,ind]=np.power(X[:,1],ind)
    return Xpoly
    