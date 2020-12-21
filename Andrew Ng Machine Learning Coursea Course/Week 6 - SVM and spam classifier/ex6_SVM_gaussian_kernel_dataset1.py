# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:11:13 2020

@author: richard_dev

Summary: This script implements a support vector machine using a Gaussian
kernel on a sample dataset.
"""

#Clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC

#===================#
#=== Model setup ===#
#===================#

isplot=1
npoints=1000

#====================#
#=== Prepare data ===#
#====================#

#Get data
ex6data2=loadmat('ex6data2.mat')

#Separate into variables
Xtrain=ex6data2['X']
ytrain=ex6data2['y']

#Reformat variables
ytrain=ytrain[:,0]
Xtrainzeros=Xtrain[ytrain==0,:]
Xtrainones=Xtrain[ytrain==1,:]

#Plot data
if isplot == 1:
    plt.figure(1)
    fig1=plt.scatter(Xtrainzeros[:,0],Xtrainzeros[:,1],marker="o",color='b')
    fig1=plt.scatter(Xtrainones[:,0],Xtrainones[:,1],marker="o",color='r')
    
#===============#
#=== Run SVM ===#
#===============#

#Run SVM
clf=SVC(C=100000,kernel='rbf',gamma='scale')
clf.fit(Xtrain,ytrain)

#Get model attributes
supportvectors=clf.support_vectors_
coefficients=clf.dual_coef_[0]
intercept=clf.intercept_[0]

#Plot support vectors
fig1=plt.scatter(supportvectors[:,0],supportvectors[:,1],marker="x",color='g')

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
    Xplot=np.array([xx1.ravel(),xx2.ravel()]).T
    Z=clf.decision_function(Xplot)
        
    #Interpret value of Z
    Z[Z>=0]=1
    Z[Z<0]=0
    Z=Z.reshape(np.shape(xx1))
    
    #Plot contour
    fig2=plt.contour(xx1,xx2,Z)

    #Plot scatter
    fig2=plt.scatter(Xtrainzeros[:,0],Xtrainzeros[:,1],marker="o")
    fig2=plt.scatter(Xtrainones[:,0],Xtrainones[:,1],marker="+")
