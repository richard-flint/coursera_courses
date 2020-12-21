# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:43:19 2020

@author: richard_dev

Summary: This script uses k-means clustering to compress a photo of a bird. It
reduces the number of colours from 255 to the number specified by the user.
"""

#Clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

#Imports
import numpy as np
import matplotlib.pyplot as plt
import ex7_k_means_clustering_functions as udf
import imageio

#===================#
#=== Model setup ===#
#===================#

isplot=1
ncentroids=16
number_of_iterations=100

#====================#
#=== Prepare data ===#
#====================#

#Get data
image = imageio.imread('bird_small.png')

#View image
fig0,ax0=plt.subplots()
ax0.imshow(image)

#Calculate additional model parameters
m=np.shape(image)[0]*np.shape(image)[1]
n=np.shape(image)[2]


#Unravel data into features
Xtrain=np.zeros([m,n])
for ind in range(n):
    Xtrain[:,ind]=image[:,:,ind].ravel()

#Plot data
if isplot == 1:
    #Create plots
    fig1,axs1=plt.subplots(3,1)
    
    #Populate scatter plots with different combinations
    axs1[0].scatter(Xtrain[:,0],Xtrain[:,1],marker=",",s=1)
    axs1[1].scatter(Xtrain[:,0],Xtrain[:,2],marker=",",s=1)
    axs1[2].scatter(Xtrain[:,1],Xtrain[:,2],marker=",",s=1)
    
    #Create 3D plot
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(Xtrain[:,0],Xtrain[:,1],Xtrain[:,2],marker=",",s=1)
    
#Initialise centroids
Xcentroids=udf.create_random_centroids(Xtrain,ncentroids,n)

#Plot centroids
if isplot == 1:
    
    #Plot in 2D
    axs1[0].scatter(Xcentroids[:,0],Xcentroids[:,1],marker="x",s=1,color='r')
    axs1[1].scatter(Xcentroids[:,0],Xcentroids[:,2],marker="x",s=1,color='r')
    axs1[2].scatter(Xcentroids[:,1],Xcentroids[:,2],marker="x",s=1,color='r')
    
    #Plot in 3D
    ax2.scatter(Xcentroids[:,0],Xcentroids[:,1],Xcentroids[:,2],marker="+",color='r')

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
    
#=== Compress image ===#
    
#Initialise variables
Xcompress=np.zeros(np.shape(Xtrain))
image_compressed=np.zeros(np.shape(image),dtype='uint8')
Xcentroids_rounded=np.round(Xcentroids,0).astype('uint8')
    
#Map each point to location of closest centroid
for ind in range(m):
    Xcompress[ind,:]=Xcentroids_rounded[closest_centroids[ind],:]
    
#Roll back up into image
for ind in range(n):
    image_compressed[:,:,ind]=np.reshape(Xcompress[:,ind],\
                [np.shape(image_compressed)[0],np.shape(image_compressed)[1]])

#View compressed image
image_compressed=imageio.core.util.Array(image_compressed)
fig7,ax7=plt.subplots()
ax7.imshow(image_compressed)

#====================#
#=== Plot results ===#
#====================#

#Plot final centroids
if isplot == 1:
    
    #2D
    
    #Create plots
    fig3,axs3=plt.subplots(3,1)
    
    #Plot points (coloured according to centroid allocation)
    for ind in range(ncentroids):
        Xtrainsubset_indices=np.where(closest_centroids==ind)[0]
        Xtrainsubset=Xtrain[Xtrainsubset_indices,:]
        axs3[0].scatter(Xtrainsubset[:,0],Xtrainsubset[:,1],marker=",",s=1)
        axs3[1].scatter(Xtrainsubset[:,0],Xtrainsubset[:,2],marker=",",s=1)
        axs3[2].scatter(Xtrainsubset[:,1],Xtrainsubset[:,2],marker=",",s=1)

    #Plot centroids
    axs3[0].scatter(Xcentroids[:,0],Xcentroids[:,1],marker="x",s=1,color='r')
    axs3[1].scatter(Xcentroids[:,0],Xcentroids[:,2],marker="x",s=1,color='r')
    axs3[2].scatter(Xcentroids[:,1],Xcentroids[:,2],marker="x",s=1,color='r')
    
    #3D

    #Create 3D plot
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111, projection='3d')
    
    #Plot points (coloured according to centroid allocation)
    for ind in range(ncentroids):
        Xtrainsubset_indices=np.where(closest_centroids==ind)[0]
        Xtrainsubset=Xtrain[Xtrainsubset_indices,:]
        ax4.scatter(Xtrainsubset[:,0],Xtrainsubset[:,1],Xtrainsubset[:,2],marker=",",s=1)
    
    #Plot centroids
    ax4.scatter(Xcentroids[:,0],Xcentroids[:,1],Xcentroids[:,2],marker="+",color='r')
    
#Plot movement of centroids
if isplot == 1:
    
    #2D
    
    #Create plot
    fig5,axs5=plt.subplots(3,1)

    #Plot points (coloured according to centroid allocation)
    for ind in range(ncentroids):
        Xtrainsubset_indices=np.where(closest_centroids==ind)[0]
        Xtrainsubset=Xtrain[Xtrainsubset_indices,:]
        axs5[0].scatter(Xtrainsubset[:,0],Xtrainsubset[:,1],marker=",",s=1)
        axs5[1].scatter(Xtrainsubset[:,0],Xtrainsubset[:,2],marker=",",s=1)
        axs5[2].scatter(Xtrainsubset[:,1],Xtrainsubset[:,2],marker=",",s=1)

    #Plot start and end centroids
    Xcentroids_start=np.reshape(Xcentroids_history[0,:],np.shape(Xcentroids))
    Xcentroids_end=np.reshape(Xcentroids_history[ncentroids,:],np.shape(Xcentroids))
    
    axs5[0].scatter(Xcentroids_start[:,0],Xcentroids_start[:,1],marker="+",color='r')
    axs5[1].scatter(Xcentroids_start[:,0],Xcentroids_start[:,2],marker="+",color='r')
    axs5[2].scatter(Xcentroids_start[:,1],Xcentroids_start[:,2],marker="+",color='r')
    
    axs5[0].scatter(Xcentroids_end[:,0],Xcentroids_end[:,1],marker="+",color='g')
    axs5[1].scatter(Xcentroids_end[:,0],Xcentroids_end[:,2],marker="+",color='g')
    axs5[2].scatter(Xcentroids_end[:,1],Xcentroids_end[:,2],marker="+",color='g')
    
    #Plot line inbetween
    for ind in range(ncentroids):
        axs5[0].plot(Xcentroids_history[:,ind*n],Xcentroids_history[:,ind*n+1],color='k')
        axs5[1].plot(Xcentroids_history[:,ind*n],Xcentroids_history[:,ind*n+2],color='k')
        axs5[2].plot(Xcentroids_history[:,ind*n+1],Xcentroids_history[:,ind*n+2],color='k')
        
    #3D
        
    #Create 3D plot
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111, projection='3d')
    
    #Plot points (coloured according to centroid allocation)
    for ind in range(ncentroids):
        Xtrainsubset_indices=np.where(closest_centroids==ind)[0]
        Xtrainsubset=Xtrain[Xtrainsubset_indices,:]
        ax6.scatter(Xtrainsubset[:,0],Xtrainsubset[:,1],Xtrainsubset[:,2],marker=",",s=1)
        
    #Plot start and end centroids
    Xcentroids_start=np.reshape(Xcentroids_history[0,:],np.shape(Xcentroids))
    Xcentroids_end=np.reshape(Xcentroids_history[ncentroids,:],np.shape(Xcentroids))
    
    ax6.scatter(Xcentroids_start[:,0],Xcentroids_start[:,1],Xcentroids_start[:,2],marker="+",color='r')
    ax6.scatter(Xcentroids_end[:,0],Xcentroids_end[:,1],Xcentroids_end[:,2],marker="+",color='g')

    #Plot line inbetween
    for ind in range(ncentroids):
        ax6.plot(Xcentroids_history[:,ind*n],Xcentroids_history[:,ind*n+1],\
                 Xcentroids_history[:,ind*n+2],color='k')