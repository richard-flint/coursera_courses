# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:05:30 2020

@author: richard_dev

Summary: This script implements a support vector machine using a Gaussian
kernel on a second sample dataset.
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
sigma=np.array([0.01,0.03,0.1,0.3,1,3,10,30,100,300])
C=np.array([0.01,0.03,0.1,0.3,1,3,10,30,100,300])

#====================#
#=== Prepare data ===#
#====================#

#Get data
ex6data3=loadmat('ex6data3.mat')

#Separate into variables
Xtrain=ex6data3['X']
ytrain=ex6data3['y']
Xval=ex6data3['Xval']
yval=ex6data3['yval']

#Reformat variables
ytrain=ytrain[:,0]
yval=yval[:,0]
Xtrainzeros=Xtrain[ytrain==0,:]
Xtrainones=Xtrain[ytrain==1,:]
Xvalzeros=Xval[yval==0,:]
Xvalones=Xval[yval==1,:]

#Plot data
if isplot == 1:
    fig1,ax1=plt.subplots()
    ax1.scatter(Xtrainzeros[:,0],Xtrainzeros[:,1],marker="o",color='b')
    ax1.scatter(Xtrainones[:,0],Xtrainones[:,1],marker="o",color='r')
    
    fig2,ax2=plt.subplots()
    ax2.scatter(Xvalzeros[:,0],Xvalzeros[:,1],marker="o",color='b')
    ax2.scatter(Xvalones[:,0],Xvalones[:,1],marker="o",color='r')
    
#Calculate further model parameters
n_sigma=len(sigma)
n_C=len(C)

#Initialise further model vectors
SCM_performance=np.zeros([n_sigma,n_C],dtype='float64')

#===============#
#=== Run SVM ===#
#===============#

#Iterate over multiple values of sigma and C
for inds in range(n_sigma):
    #print(inds)
    for indc in range (n_C):

        #Run SVM
        clf=SVC(C=C[indc],kernel='rbf',gamma=sigma[inds])
        clf.fit(Xtrain,ytrain)
        
        #Get model attributes
        supportvectors=clf.support_vectors_
        coefficients=clf.dual_coef_[0]
        intercept=clf.intercept_[0]
                
        #Calculate performance using the validation set
        SCM_performance[inds,indc]=clf.score(Xval,yval)

#Plot SCM performance
fig3,ax3=plt.subplots()
xx1,xx2=np.meshgrid(sigma,C)
ax3.contourf(np.linspace(1,10,n_sigma),np.linspace(1,10,n_C),SCM_performance)

#Find best performance
best_performance=np.max(SCM_performance)
sigma_index=np.where(SCM_performance==best_performance)[0]
C_index=np.where(SCM_performance==best_performance)[1]

n_pairs=len(sigma_index)
sigma_C_pairs=np.zeros([n_pairs,2])
for ind in range(n_pairs):
    sigma_C_pairs[ind,0]=sigma[sigma_index[ind]]
    sigma_C_pairs[ind,1]=C[C_index[ind]]
    
#Run and plot for best performance
if isplot == 1:
    
    #Setup plot on training data
    fig4,ax4=plt.subplots()
    
    x0=np.ones(npoints)
    x1=np.linspace(np.min(Xtrain[:,0]),np.max(Xtrain[:,0]),npoints)
    x2=np.linspace(np.min(Xtrain[:,1]),np.max(Xtrain[:,1]),npoints)
    xx1train,xx2train=np.meshgrid(x1,x2)
    
    #Setup plot on validation data
    fig5,ax5=plt.subplots()
    
    x0=np.ones(npoints)
    x1=np.linspace(np.min(Xval[:,0]),np.max(Xval[:,0]),npoints)
    x2=np.linspace(np.min(Xval[:,1]),np.max(Xval[:,1]),npoints)
    xx1val,xx2val=np.meshgrid(x1,x2)
    
    #Loop over all best performing pairs
    for index in range(n_pairs):
    
        #Run SVM again
        clf=SVC(C=sigma_C_pairs[index,1],kernel='rbf',gamma=sigma_C_pairs[index,0])
        clf.fit(Xtrain,ytrain)
        
        #Get h values for meshgrid (2 ways of doing this)
        Xplottrain=np.array([xx1train.ravel(),xx2train.ravel()]).T
        Ztrain=clf.decision_function(Xplottrain)
        Xplotval=np.array([xx1val.ravel(),xx2val.ravel()]).T
        Zval=clf.decision_function(Xplotval)
            
        #Interpret value of Z
        Ztrain[Ztrain>=0]=1
        Ztrain[Ztrain<0]=0
        Ztrain=Ztrain.reshape(np.shape(xx1train))
        
        Zval[Zval>=0]=1
        Zval[Zval<0]=0
        Zval=Zval.reshape(np.shape(xx1val))
        
        #Plot contour
        ax4.contour(xx1train,xx2train,Ztrain)
        ax5.contour(xx1val,xx2val,Zval)
        
    #Plot scatter
    ax4.scatter(Xtrainzeros[:,0],Xtrainzeros[:,1],marker="o")
    ax4.scatter(Xtrainones[:,0],Xtrainones[:,1],marker="+")
    ax5.scatter(Xvalzeros[:,0],Xvalzeros[:,1],marker="o")
    ax5.scatter(Xvalones[:,0],Xvalones[:,1],marker="+")