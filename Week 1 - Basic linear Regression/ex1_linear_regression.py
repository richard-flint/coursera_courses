# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:00:02 2020

@author: richa
"""

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics
import ex1_linear_regression_functions as udf

#Initialise key variables
alpha=0.02
number_of_iterations=1000
J_history=np.zeros(number_of_iterations+1)
theta_history=np.zeros([number_of_iterations+1,2])

#Load data into Python
data = pd.read_csv('ex1data1.txt', sep=",", header=None,names=["x1","y"])

#Get number of rows
m=np.shape(data)[0]

#Add x0 column to data
x0=np.ones((m,1), dtype=int)
data.insert(0,"x0",x0,True)

#Define X and y arrays from the dataframe
X=data[['x0','x1']].values
x1=data['x1'].values
X_transpose=X.conj().transpose()
y=data['y'].values

#Plot data
plt.figure(1)
plt.plot(x1, y,'o')

#Inialise theta
theta=np.array([0,0],dtype=np.float64)

#Calculate initial cost
J_history[0]=udf.compute_cost(X_transpose,y,theta,m)
theta_history[[0,0]]=theta[0]
theta_history[[0,1]]=theta[1]

#Iterate over theta
for i in range(number_of_iterations):
    
    #Compute the hypothesis
    h=np.dot(theta,X_transpose)
    
    #Calculate new theta
    theta[0]=theta[0]-(alpha/m)*np.dot((h-y),x0)
    theta[1]=theta[1]-(alpha/m)*np.dot((h-y),x1)
    
    #Save values of J to graph
    J_history[i+1]=udf.compute_cost(X_transpose,y,theta,m)
    theta_history[[i+1,0]]=theta[0]
    theta_history[[i+1,1]]=theta[1]
    
#Plot final regression line
plt.figure(1)
plt.plot(x1, y,'o')
plot_x=np.linspace(5, max(x1),2)
plot_y=theta[0]+theta[1]*plot_x
plt.plot(plot_x,plot_y)
    
#plot cost function over multiple iterations
plt.figure(2)
plot_x = np.linspace(0, number_of_iterations,number_of_iterations+1)
plot_y = J_history
plt.semilogy(plot_x, plot_y)

print("Theta 0 = ",theta[0],"\nTheta 1 = ",theta[1])

#Plot cost funcction
number_of_points=1000
plot_range=5
cost_contours=np.zeros((number_of_points,number_of_points))
theta_0=np.linspace(-plot_range,plot_range,number_of_points)
theta_1=np.linspace(-plot_range,plot_range,number_of_points)
x_plot,y_plot=np.meshgrid(theta_0,theta_1)
theta=np.array([0,0],dtype=np.float64)
for i in range(number_of_points):
    theta[0]=theta_0[i]
    for j in range(number_of_points):
        theta[1]=theta_1[j]
        J=udf.compute_cost(X_transpose,y,theta,m)
        cost_contours[i,j]=udf.compute_cost(X_transpose,y,theta,m)
z_plot=cost_contours

#Surface plot
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot,y_plot,z_plot)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.contour(theta_0,theta_1,z_plot)
line_x=theta_history[:,0]
line_y=theta_history[:,1]
ax.plot(line_x,line_y)