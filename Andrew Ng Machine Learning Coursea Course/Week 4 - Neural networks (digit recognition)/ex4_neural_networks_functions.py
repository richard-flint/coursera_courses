# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:05:00 2020

@author: richard_dev
"""

import numpy as np

def create_binary_y_variable(y,i):
    y_new=np.copy(y)
    y_new[y==i+1]=1
    y_new[y!=i+1]=0
    return y_new

def sigmoid_function(z):
    g=1/(1+np.exp(-z))
    return g

def compute_cost(h,Y,theta1,theta2,regularisation,k,m):
       
    #Compute the cost                           #Y=[5000,10]
    J1=-np.multiply(Y,np.log(h))                #J1=[5000,10]
    J2=-np.multiply((1-Y),np.log(1-h))          #J2=[5000,10]
    J=J1+J2                                     #J=[5000,10]
    J=(1/m)*np.sum(np.sum(J,1))
    
    if regularisation == 1:
        R1=np.sum(np.sum(np.multiply(theta1[:,1:],theta1[:,1:]),1))
        R2=np.sum(np.sum(np.multiply(theta2[:,1:],theta2[:,1:]),1))
        R=(k/(2*m))*(R1+R2)
        J=J+R
        
    #Return value
    return J

def gradient_check(h,Y,theta1,theta2,regularisation,k,m,epsilon_grad_check,D1,D2):
    
    #Initialise variables
    J_theta1=np.zeros([np.shape(theta1)[0],np.shape(theta1)[1]])
    J_theta2=np.zeros([np.shape(theta2)[0],np.shape(theta2)[1]])
    
    #Vary over theta1
    for i in range(np.shape(theta1)[0]):
        for j in range(np.shape(theta1)[1]):
            
            #Define theta with +/- epsilon
            theta1_minus=np.copy(theta1)
            theta1_plus=np.copy(theta1)
            
            theta1_minus[i,j]=theta1_minus[i,j]-epsilon_grad_check
            theta1_plus[i,j]=theta1_plus[i,j]+epsilon_grad_check
            
            #Compute cost with +/- epsilon
            J_minus=compute_cost(h,Y,theta1_minus,theta2,regularisation,k,m)
            J_plus=compute_cost(h,Y,theta1_plus,theta2,regularisation,k,m)
            
            #Save the results
            J_theta1[i,j]=(J_plus-J_minus)/(2*epsilon_grad_check)
    
    #Vary over theta2
    for i in range(np.shape(theta2)[0]):
        for j in range(np.shape(theta2)[1]):
            
            #Define theta with +/- epsilon
            theta2_minus=np.copy(theta2)
            theta2_plus=np.copy(theta2)
            
            theta2_minus[i,j]=theta2_minus[i,j]-epsilon_grad_check
            theta2_plus[i,j]=theta2_plus[i,j]+epsilon_grad_check
            
            #Compute cost with +/- epsilon
            J_minus=compute_cost(h,Y,theta1,theta2_minus,regularisation,k,m)
            J_plus=compute_cost(h,Y,theta1,theta2_plus,regularisation,k,m)
            
            #Save the results
            J_theta2[i,j]=(J_plus-J_minus)/(2*epsilon_grad_check)
            
    
    #Calculate difference with backwards propagation
    D1_difference=J_theta1-D1
    D2_difference=J_theta2-D2
    max_difference1=np.max(abs(D1_difference))
    max_difference2=np.max(abs(D2_difference))
    
    #Print results
    print("Max difference 1: ",max_difference1)
    print("Max difference 2: ",max_difference2)
    
    return D1_difference,D2_difference

def feed_forward(X,theta1,theta2):
    
    #Compute a1
    a1=np.copy(X)                           #a1=[5000,401]
                                            #a1.T=[401,5000]
    #Compute a2
    z2=np.dot(theta1,a1.T)                              #z2=[25,5000]
    a2=sigmoid_function(z2)                         #a2=[25,5000]
    a2=a2.T                                             #a2=[5000,25]
    a2=np.hstack([np.ones([np.shape(a2)[0],1]),a2])     #a2=[5000,26]
                                                        #Add bias layer
    
    #Compute a3
    z3=np.dot(theta2,a2.T)                  #z3=[10,5000]
    a3=sigmoid_function(z3)             #a3=[10,5000]
    a3=a3.T                                 #a3=[5000,10]
    
    #Compute h
    h=np.copy(a3)                           #h=[5000,10]
    
    return h,a1,a2,a3