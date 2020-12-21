# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:33:25 2020

@author: richard_dev

Summary: This is a simple clustering k-means algorithm that uses a sample
dataset.
"""

#Clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import ex7_k_means_clustering_functions as udf

#===================#
#=== Model setup ===#
#===================#

isplot=1
ncentroids=3
number_of_iterations=100
dataset=2

#====================#
#=== Prepare data ===#
#====================#

#Get data
if dataset == 1:
    ex7data=loadmat('ex7data1.mat')
elif dataset == 2:
    ex7data=loadmat('ex7data2.mat')
#ex7faces=loadmat('ex7faces.mat')

#Separate into variables
Xtrain=ex7data['X']

#Calculate additional model parameters
m=np.shape(Xtrain)[0]
n=np.shape(Xtrain)[1]

#Plot data
if isplot == 1:
    #Create plots
    fig1,ax1=plt.subplots()
    #fig2,ax2=plt.subplots()
    #fig3,ax3=plt.subplots()
    
    ax1.scatter(Xtrain[:,0],Xtrain[:,1],marker="o")
    #ax2.scatter(X2train[:,0],X2train[:,1],marker="o")
    #ax3.scatter(Xftrain[:,0],Xftrain[:,1],marker="o")
    
#Initialise centroids
Xcentroids=udf.create_random_centroids(Xtrain,ncentroids,n)

#Plot centroids
if isplot == 1:
    ax1.scatter(Xcentroids[:,0],Xcentroids[:,1],marker="+")

#=============================#
#=== Run K-means algorithm ===#
#=============================#

#Initialise variables
Xcentroids_history=np.zeros([number_of_iterations+1,ncentroids*n])
Xcentroids_history[0,:]=Xcentroids.ravel()

#=== Check each centroid ===#
    
#Loop until each centroid has at least one point closest to it
while True:
    
    #Set flag for do-while loop
    test_check=0
    
    #Calculate the distance between each point and each centroid
    distances_to_centroids=udf.calculate_distances_to_centroids(Xtrain,Xcentroids,m,ncentroids)
    
    #Find the closest centroid to each point
    closest_centroids=udf.find_closest_centroids(distances_to_centroids)
    
    #Check that there is at least one point closest to each centroid
    Xcentroids,test_check=udf.check_centroids(Xtrain,Xcentroids,closest_centroids,ncentroids,n)
            
    #If we have relocated one point, then restart the iterations
    if test_check==0:
        break
    
#=== Run algorithm ===#
for ind in range(number_of_iterations):
    
    #Calculate the distance between each point and each centroid
    distances_to_centroids=udf.calculate_distances_to_centroids(Xtrain,Xcentroids,m,ncentroids)
    
    #Find the closest centroid to each point
    closest_centroids=udf.find_closest_centroids(distances_to_centroids)
    
    #Find new centroid position (average locations of points closest to centroid)
    Xcentroids=udf.find_new_centroid_positions(Xtrain,Xcentroids,ncentroids,closest_centroids)
        
    #Save centroid positions for each iteration
    Xcentroids_history[ind+1,:]=Xcentroids.ravel()

#Plot final centroids
if isplot == 1:
    #Create plots
    fig2,ax2=plt.subplots()
    
    #Plot points
    ax2.scatter(Xtrain[:,0],Xtrain[:,1],marker="o")

    #Plot centroids
    ax2.scatter(Xcentroids[:,0],Xcentroids[:,1],marker="+")
    
#Plot movement of centroids
if isplot == 1:
    #Create plot
    fig3,ax3=plt.subplots()

    #Plot points (coloured according to centroid allocation)
    for ind in range(ncentroids):
        Xtrainsubset_indices=np.where(closest_centroids==ind)[0]
        Xtrainsubset=Xtrain[Xtrainsubset_indices,:]
        ax3.scatter(Xtrainsubset[:,0],Xtrainsubset[:,1],marker="o")

    #Plot start and end centroids
    Xcentroids_start=np.reshape(Xcentroids_history[0,:],np.shape(Xcentroids))
    Xcentroids_end=np.reshape(Xcentroids_history[ncentroids,:],np.shape(Xcentroids))
    ax3.scatter(Xcentroids_start[:,0],Xcentroids_start[:,1],marker="+",color='r')
    ax3.scatter(Xcentroids_end[:,0],Xcentroids_end[:,1],marker="+",color='g')
    
    #Plot line inbetween
    for ind in range(ncentroids):
        ax3.plot(Xcentroids_history[:,ind*n],Xcentroids_history[:,ind*n+1],color='k')