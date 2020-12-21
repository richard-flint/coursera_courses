# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:14:58 2020

@author: richard_dev
"""
import numpy as np

def create_random_centroids(Xtrain,ncentroids,n):
    Xcentroids=np.random.rand(ncentroids,n)
    for ind in range(n):
        Xcentroids[:,ind]=Xcentroids[:,ind]*(np.max(Xtrain[:,ind])-np.min(Xtrain[:,ind]))+1
    return Xcentroids

def calculate_distances_to_centroids(Xtrain,Xcentroids,m,ncentroids):
    distances_to_centroids=np.zeros([m,ncentroids])
    for ind in range(ncentroids):
        Xtemp=Xtrain-Xcentroids[ind,:]
        distances_to_centroids[:,ind]=np.sum(np.multiply(Xtemp,Xtemp),1)
    return distances_to_centroids

def find_closest_centroids(distances_to_centroids):
    min_distances_to_centroids=np.array([np.min(distances_to_centroids,1)]).T
    closest_centroids=np.where(distances_to_centroids==min_distances_to_centroids)[1]
    return closest_centroids

def check_centroids(Xtrain,Xcentroids,closest_centroids,ncentroids,n):
    #Create array to store tests
    test_check_history=np.zeros(ncentroids)
    
    #Run test for each centroid
    for ind in range(ncentroids):
        
        #Check if there is at least one point whose closest centroid is this centroid
        test=np.isin(ind,closest_centroids)
        
        #If the centroid does not have a closest point, relocate centroid
        if test==False:
            Xcentroids[ind,:]=create_random_centroids(Xtrain,1,n)
            test_check_history[ind]=1
    
    #If at least one centroid has been reset, then flag this
    if np.sum(test_check_history)>0:
        test_check=1
    elif np.sum(test_check_history)==0:
        test_check=0
    
    #Return values
    return Xcentroids,test_check

def find_new_centroid_positions(Xtrain,Xcentroids,ncentroids,closest_centroids):
    
    #Loop over centroids
    for ind in range(ncentroids):
        
        #Find indices of points closest to centroid in X matrix
        indices_of_points_closest_to_centroid=np.where(closest_centroids==ind)
        
        #Find position of points closest to centroid
        position_of_points_closest_to_centroid=\
            Xtrain[indices_of_points_closest_to_centroid[0],:]
            
        #Find average position of points closest to centroid
        average_position_of_points_closest_to_centroid=\
            np.sum(position_of_points_closest_to_centroid,0)\
            /np.shape(position_of_points_closest_to_centroid)[0]
        
        #Save average position as new centroid position
        Xcentroids[ind,:]=average_position_of_points_closest_to_centroid
        
    #Return new centroid positions
    return Xcentroids