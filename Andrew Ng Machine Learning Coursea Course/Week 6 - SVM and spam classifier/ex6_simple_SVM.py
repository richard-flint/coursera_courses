# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:52:30 2020

@author: richard_dev

Summary: This script implements a simple support vector machine (SVM) on a 
sample dataset.
"""

#Clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import LinearSVC

#===================#
#=== Model setup ===#
#===================#

isplot=1
npoints=1000
find_line=2

#====================#
#=== Prepare data ===#
#====================#

#Get data
ex6data1=loadmat('ex6data1.mat')
ex6data2=loadmat('ex6data2.mat')
ex6data3=loadmat('ex6data3.mat')

#Separate into variables
Xtrain=ex6data1['X']
ytrain=ex6data1['y']

#Reformat variables
ytrain=ytrain[:,0]
Xtrainzeros=Xtrain[ytrain==0,:]
Xtrainones=Xtrain[ytrain==1,:]

#Plot data
if isplot == 1:
    plt.figure(1)
    fig1=plt.scatter(Xtrainzeros[:,0],Xtrainzeros[:,1],marker="o")
    fig1=plt.scatter(Xtrainones[:,0],Xtrainones[:,1],marker="+")

#===============#
#=== Run SVM ===#
#===============#

#Run SVM
clf=LinearSVC(C=1000)
clf.fit(Xtrain,ytrain)

#Get model attributes
theta=np.array([clf.intercept_[0],clf.coef_[0,0],clf.coef_[0,1]])

#Plot SVM
if isplot == 1:
    #Set up plot
    plt.figure(2)
    
    #Set up meshgrid
    x0=np.ones(npoints)
    x1=np.linspace(np.min(Xtrain[:,0]),np.max(Xtrain[:,0]),npoints)
    x2=np.linspace(np.min(Xtrain[:,1]),np.max(Xtrain[:,1]),npoints)
    xx1,xx2=np.meshgrid(x1,x2)
    
    #Get h values for meshgrid (2 ways of doing this)
    if find_line==1:
        Xplot=np.array([xx1.ravel(),xx2.ravel()]).T
        Z=clf.decision_function(Xplot)
    elif find_line==2:
        xx0=np.ones(np.shape(xx1.ravel()))
        Xplot=np.array([xx0,xx1.ravel(),xx2.ravel()]).T
        Z=np.dot(Xplot,theta)
    
    #Interpret value of Z
    Z[Z>=0]=1
    Z[Z<0]=0
    Z=Z.reshape(np.shape(xx1))
    
    #Plot contour
    fig2=plt.contour(xx1,xx2,Z)

    #Plot scatter
    fig2=plt.scatter(Xtrainzeros[:,0],Xtrainzeros[:,1],marker="o")
    fig2=plt.scatter(Xtrainones[:,0],Xtrainones[:,1],marker="+")
