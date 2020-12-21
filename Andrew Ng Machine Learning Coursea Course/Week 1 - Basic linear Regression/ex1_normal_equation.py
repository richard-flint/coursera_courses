# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:33:12 2020

@author: richa

Summary: This script calculates a simple linear regression using the normal
equation.
"""

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Initialise key variables
alpha=0.02
number_of_iterations=1500
J_history=np.zeros(number_of_iterations+1)

#Load data into Python
data = pd.read_csv('ex1data1.txt', sep=",", header=None,names=["x1","y"])

#Get number of rows
data_size=np.shape(data)
m=data_size[0]

#Add x0 column to data
x0_temp=np.ones((m,1), dtype=int)
data.insert(0,"x0",x0_temp,True)

#Define X and y arrays from the dataframe
X=data[['x0','x1']].values
x0=data['x0'].values
x1=data['x1'].values
y=data['y'].values

#Plot data
plt.style.use('seaborn-whitegrid')
plt.plot(x1, y,'o')

#Calculate theta
a=np.linalg.inv(np.dot(X.transpose(),X))
b=np.dot(X.transpose(),y)
theta=np.dot(a,b)
print("theta=",theta)

#Plot final regression line
plt.plot(x1, y,'o')
plot_x=np.linspace(5, max(x1),2)
plot_y=theta[0]+theta[1]*plot_x
plt.plot(plot_x,plot_y)