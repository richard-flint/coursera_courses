# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:06:51 2020

@author: richard_dev

Summary: This script implements a neural network algorithm to identify the
digits 0 to 9 in greyscale images. Note that This implementation uses 
pre-defined Theta values provided by course documentation
"""

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
import ex3_functions as udf

#Get data
ex3data1=loadmat('ex3data1.mat')
ex3weights=loadmat('ex3weights.mat')

#Unpack data
X=ex3data1['X']
y=ex3data1['y']
Theta1=ex3weights['Theta1']
Theta2=ex3weights['Theta2']

#Redefine the 0 value in the y data
y=y[:,0]
y[y==10]=0

#Add x0 row to X
x0=np.ones(np.shape(X)[0])
X=np.column_stack([x0,X])

#Define data characteristics
m=np.shape(X)[0]
n=np.shape(X)[1]

#****************************************************************************#
#********************************** ML Code *********************************#
#****************************************************************************#

#Layer 1
a1=udf.sigmoid_function(np.dot(Theta1,X.transpose()))
a0=np.ones([1,m])
a1=np.row_stack([a0,a1])

#Layer2
a2=udf.sigmoid_function(np.dot(Theta2,a1))
a2=a2.transpose()

#Estimated numbers
max_per_row=np.amax(abs(a2),1)
estimated_numbers=np.zeros(m)
for i in range(m):
    estimated_number=np.where(abs(a2[i,:])==max_per_row[i])
    estimated_numbers[i]=estimated_number[0].item()
estimated_numbers=estimated_numbers+1
estimated_numbers[estimated_numbers==10]=0
    
#Calculate percentage of correct answers
correct_numbers=(estimated_numbers==y)*1
percentage_correct=(sum(correct_numbers)/m)*100
print('Percentage correct = ',percentage_correct,'%')