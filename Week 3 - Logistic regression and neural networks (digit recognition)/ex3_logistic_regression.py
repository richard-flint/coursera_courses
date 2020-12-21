#Clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

#Imports
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import ex3_logistic_regression_functions as udf

#Define if regularisation is on or off
regularisation = 1    #0=OFF, 1=ON

#Get data
ex3data1=loadmat('ex3data1.mat')
ex3weights=loadmat('ex3weights.mat')

#Unpack data
X=ex3data1['X']
y=ex3data1['y']

#Redefine the 0 value in the y data
y=y[:,0]
y[y==10]=0

#Add x0 row to X
x0=np.ones(np.shape(X)[0])
X=np.column_stack([x0,X])

#Define data characteristics
m=np.shape(X)[0]
n=np.shape(X)[1]

#Plot X
# =============================================================================
# plt.figure(1)
# x1=np.linspace(1,n,n)
# x2=np.linspace(1,m,m)
# plt.contourf(x1,x2,X)
# =============================================================================

#Plot individual numbers
row_num=2050
x=np.reshape(X[row_num,1:],[20,20])
x1=np.linspace(1,20,20)
x2=np.linspace(1,20,20)
plt.contourf(x1,x2,x,cmap='gray_r')

#Plot y
plt.figure(2)
fig2=plt.plot(y)

#Redefine y data
#This creates a two dimensional Y matrix where each column is a different
#number from 0 to 9. This is used in the 1-vs-all approach implemented below.
Y=np.zeros([m,10])
for i in range(10):
    Y[:,i]=udf.create_binary_y_variable(y,i)


#****************************************************************************#
#********************************** ML Code *********************************#
#****************************************************************************#
#Define ML variables
Theta=np.zeros([n,10],dtype='float64')
number_of_iterations=1000
alpha=0.25
k=1
J_history=np.zeros([number_of_iterations+1,10],dtype='float64')
theta_history=np.zeros([number_of_iterations+1,n,10],dtype='float64')

# 1 vs all iterate over numbers 0 to 9
for i in range(10):
    
    #Extract relevant value of y and theta
    y_bin=Y[:,i]
    theta=Theta[:,i]
    
    #Calculate initial cost
    if regularisation == 0:
        J=udf.compute_J(X,y_bin,theta,m)
    elif regularisation == 1:
        J=udf.compute_J_with_regularisation(X,y_bin,theta,m,k)
    J_history[0,i]=J
    theta_history[0,:,i]=theta
    
    #Iterate
    for j in range(number_of_iterations):
        
        #Update theta
        if regularisation == 0:
            theta=udf.update_theta(X,y_bin,theta,alpha,m)
        elif regularisation == 1:
            theta=udf.update_theta_with_regularisation(X,y_bin,theta,alpha,m,k)
        
        #Save value of theta in theta_history
        theta_history[j+1,:,i]=theta
        
        #Calculate and save cost to J_history
        if regularisation == 0:
            J=udf.compute_J(X,y_bin,theta,m)
        elif regularisation == 1:
            J=udf.compute_J_with_regularisation(X,y_bin,theta,m,k)
        J_history[j+1,i]=J
    
    #Update Theta matrix
    Theta[:,i]=theta

#Review outputs
plt.figure(3)
plt.plot(J_history[:,0])

#****************************************************************************#
#********************************** ML Test *********************************#
#****************************************************************************#

#Estimate the number based on ML classification
x=X[row_num,:]
number=udf.test_classification(x,Theta)
print('Number estimate is:',number)
print('Actual number is: ',y[row_num].item())

#Plot the number
x=np.reshape(X[row_num,1:],[20,20])
x1=np.linspace(1,20,20)
x2=np.linspace(1,20,20)
plt.figure(5)
plt.contourf(x1,x2,x,cmap='gray_r')

#Calculate percentage of correct answers
estimated_numbers=np.zeros(m)
estimated_numbers=udf.test_classification_all(X,Theta,m)
correct_numbers=(estimated_numbers==y)*1
percentage_correct=(sum(correct_numbers)/m)*100
print('Percentage correct = ',percentage_correct,'%')




    

