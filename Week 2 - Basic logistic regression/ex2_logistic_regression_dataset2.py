# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:21:46 2020

@author: richa
"""

#Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat

#**************************#
#*** Initial data setup ***#
#**************************#

# Load data
data = pd.read_csv('ex2data2.txt',sep=",", header=None,names=["x1","x2","y"])

#Define variables
X=data[['x1','x2']].values
y=data['y'].values
x1=data['x1'].values
x2=data['x2'].values

#Add x0 to data
m=data.shape[0]
x0=np.ones(m)
X=np.stack((x0,x1,x2))
X=X.transpose()

#Feature scaling
x0_n=x0
x1_n=(x1-stat.mean(x1))/(max(x1)-min(x1))
x2_n=(x2-stat.mean(x2))/(max(x2)-min(x2))
X_transpose_n=np.array([x0_n,x1_n,x2_n])
X_n=X_transpose_n.transpose()

#Initialise theta
theta=np.zeros((28,),dtype=np.float64)

#*****************#
#*** Plot data ***#
#*****************#

#Plot data
plt.figure(1)
x1_y0=x1[y == 0]
x2_y0=x2[y == 0]
x1_y1=x1[y != 0]
x2_y1=x2[y != 0]
plt.plot(x1_y0,x2_y0,'o')
plt.plot(x1_y1,x2_y1,'x')

#Plot data with feature scaling
plt.figure(2)
x1n_y0=x1_n[y == 0]
x2n_y0=x2_n[y == 0]
x1n_y1=x1_n[y != 0]
x2n_y1=x2_n[y != 0]
plt.plot(x1n_y0,x2n_y0,'o')
plt.plot(x1n_y1,x2n_y1,'x')

#****************************************#
#*** Create polynomial feature vector ***#
#****************************************#

#Function for calculating X_p
def calculate_xp(x_0,x_1,x_2):
    X_p=np.vstack((x_0,x_1,x_2,
                   x_1**2,x_1*x_2,x_2**2,
                   x_1**3,(x_1**2)*x_2,x_1*(x_2**2),x_2**3,
                   x_1**4,(x_1**3)*x_2,(x_1**2)*(x_2**2),x_1*(x_2**3),x_2**4,
                   x_1**5,(x_1**4)*x_2,(x_1**3)*(x_2**2),(x_1**2)*(x_2**3),x_1*(x_2**4),x_2**5,
                   x_1**6,(x_1**5)*x_2,(x_1**4)*(x_2**2),(x_1**3)*(x_2**3),(x_1**2)*(x_2**4),x_1*(x_2**5),x_2**6))
    X_p=X_p.transpose()
    return X_p

#Define X and X_n for vectorised polynomial calcuation
X_p=calculate_xp(x0,x1,x2)
X_n_p=calculate_xp(x0_n,x1_n,x2_n)

#************************#
#*** Define functions ***#
#************************#

#Sigmoid function
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

#Compute hypothesis
def compute_h(X,theta):
    z=np.dot(theta,X.transpose())
    h=sigmoid(z)
    return h

#Compute cost
def compute_cost(X,y,theta,m,k):
    h=compute_h(X,theta)
    j1=(-1/m)*(np.dot(y,np.log(h))+np.dot((1-y),np.log(1-h)))
    j2=(k/2*m)*np.dot(theta,theta)
    J=j1+j2
    return J

#Gradient descent
def estimate_theta(X_p,x0,y,m,k,number_of_iterations):
    
    #Initialise variables
    theta=np.zeros((28,),dtype=np.float64)
    J_history=np.zeros(number_of_iterations+1)
    theta_history=np.zeros((number_of_iterations+1,theta.shape[0]))
    
    #Calculate initial cost
    J_history[0]=compute_cost(X_p,y,theta,m,k)

    #Iterate
    for i in range(number_of_iterations):

        #Compute the hypothesis
        h=compute_h(X_p,theta)

        #Calculate new theta
        theta[0]=theta[0]-(alpha/m)*np.dot((h-y),x0)
        theta[1:]=theta[1:]-(alpha/m)*np.dot((h-y),X_p[:,1:])-((alpha*k)/m)*theta[1:]

        #Save values of J to graph
        J_history[i+1]=compute_cost(X_p,y,theta,m,k)
        theta_history[[i+1],:]=theta

    return theta,J_history,theta_history

#*********************#
#*** Run the model ***#
#*********************#

#Define initial parameters
number_of_iterations=1000000

#Run the model without feature scaling
alpha=15
k=0
theta_1,J_history_1,theta_history_1=estimate_theta(X_p,x0,y,m,k,number_of_iterations)

#Run the model with feature scaling
alpha=15
k=0
theta_2,J_history_2,theta_history_2=estimate_theta(X_n_p,x0,y,m,k,number_of_iterations)

#********************#
#*** Plot outputs ***#
#********************#

#Plot J
plot_x=np.linspace(0, number_of_iterations,number_of_iterations+1)
plot_y1=J_history_1
plot_y2=J_history_2

plt.figure(3)
plt_v1=plt.plot(plot_x,plot_y1)
plt_v2=plt.plot(plot_x,plot_y2)
plt.legend(['v1', 'v2','v3'])

plt.figure(4)
plt.semilogy(plot_x,plot_y1)
plt.semilogy(plot_x,plot_y2)
plt.legend(['v1', 'v2','v3'])

#*** Plot the decision boundary (no feature scaling) ***#

#Define boundaries
xmin=-1
xmax=+1.2
ymin=-1
ymax=1.2
number_of_points_x=int(xmax*100+abs(xmin)*100+1)
number_of_points_y=int(ymax*100+abs(ymin)*100+1)

#Define the grid
x1_plot=np.linspace(xmin,xmax,number_of_points_x)
x2_plot=np.linspace(ymin,ymax,number_of_points_y)
x0_plot=np.ones(x1_plot.shape)
vx,vy=np.meshgrid(x1_plot,x2_plot)

#Prepare h mesh vectors
h_mesh_1=np.zeros((number_of_points_x,number_of_points_y))

#Calculate h for each point in the grid
for i in range(number_of_points_y):
    for j in range(number_of_points_x):
        x_ij=calculate_xp(1,vx[i,j],vy[i,j])
        h_mesh_1[i,j]=compute_h(x_ij,theta_1)

#Convert into binary numbers
h_mesh_1[h_mesh_1>0.5]=1
h_mesh_1[h_mesh_1<=0.5]=0

plt.figure(5)
plt.plot(x1_y0,x2_y0,'o')
plt.plot(x1_y1,x2_y1,'x')
plt.contour(vx,vy,h_mesh_1)

#Plot the decision boundary (with feature scaling)

#Define boundaries
xmin=-0.5
xmax=+0.5
ymin=-0.5
ymax=0.5
number_of_points_x=int(xmax*100+abs(xmin)*100+1)
number_of_points_y=int(ymax*100+abs(ymin)*100+1)

#Define the grid
x1_plot=np.linspace(xmin,xmax,number_of_points_x)
x2_plot=np.linspace(ymin,ymax,number_of_points_y)
x0_plot=np.ones(x1_plot.shape)
vx,vy=np.meshgrid(x1_plot,x2_plot)

#Prepare h mesh vectors
h_mesh_2=np.zeros((number_of_points_x,number_of_points_y))

#Calculate h for each point in the grid
for i in range(number_of_points_y):
    for j in range(number_of_points_x):
        x_ij=calculate_xp(1,vx[i,j],vy[i,j])
        h_mesh_2[i,j]=compute_h(x_ij,theta_2)

#Convert into binary numbers
h_mesh_2[h_mesh_2>0.5]=1
h_mesh_2[h_mesh_2<=0.5]=0

plt.figure(6)
plt.plot(x1n_y0,x2n_y0,'o')
plt.plot(x1n_y1,x2n_y1,'x')
plt.contour(vx,vy,h_mesh_2)