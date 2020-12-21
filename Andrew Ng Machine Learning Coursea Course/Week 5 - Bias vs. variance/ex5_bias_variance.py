# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:52:43 2020

@author: richard_dev

#Summary: This script investigates high variance and high bias algorithms, 
including ways to identify this in the analysis.
"""

#Clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import ex5_bias_variance_functions as udf

#===================#
#=== Model setup ===#
#===================#

#Define model parameters
isplot=0
k=0
alpha=np.array([1,1,1,1,1,1,1,1,1,1])
number_of_iterations=10000
n=8
N=10
feature_scaling=2

#====================#
#=== Prepare data ===#
#====================#

#Get data
ex5data1=loadmat('ex5data1.mat')

#Separate into variables
Xtrain=ex5data1['X']
ytrain=ex5data1['y']
ytrain=ytrain[:,0]

Xval=ex5data1['Xval']
yval=ex5data1['yval']
yval=yval[:,0]

Xtest=ex5data1['Xtest']
ytest=ex5data1['ytest']
ytest=ytest[:,0]

#Get key variables from data
mtrain=np.shape(Xtrain)[0]
mval=np.shape(Xval)[0]
mtest=np.shape(Xtest)[0]

#Add bias unit to data
x0train=np.ones([mtrain,1])
Xtrain=np.hstack([x0train,Xtrain])

x0val=np.ones([mval,1])
Xval=np.hstack([x0val,Xval])

x0test=np.ones([mtest,1])
Xtest=np.hstack([x0test,Xtest])

#Plot if on
if isplot==1:
    plt.figure(1)
    fig1=plt.scatter(Xtrain,ytrain)
    plt.figure(2)
    fig1=plt.scatter(Xtest,ytest)
    plt.figure(3)
    fig1=plt.scatter(Xval,yval)

if feature_scaling == 1:

    #Define up to N order polynomial
    Xtrainpoly=udf.create_Xpoly(Xtrain,N,mtrain)
    Xvalpoly=udf.create_Xpoly(Xval,N,mval)
    Xtestpoly=udf.create_Xpoly(Xtest,N,mtest)
    
    #Feature scaling and mean normalisation
    Xtrainpolymin=np.min(Xtrainpoly,0)
    Xtrainpolymax=np.max(Xtrainpoly,0)
    Xtrainpolyrange=Xtrainpolymax-Xtrainpolymin
    Xtrainpolymean=np.mean(Xtrainpoly,0)
    
    Xtrainpolymean[0]=0
    Xtrainpolyrange[0]=1
    
    Xtrainpolymn=np.subtract(Xtrainpoly,Xtrainpolymean)
    Xtrainpolymn=np.divide(Xtrainpolymn,Xtrainpolyrange)
    Xvalpolymn=np.subtract(Xvalpoly,Xtrainpolymean)
    Xvalpolymn=np.divide(Xvalpolymn,Xtrainpolyrange)
    Xtestpolymn=np.subtract(Xtestpoly,Xtrainpolymean)
    Xtestpolymn=np.divide(Xtestpolymn,Xtrainpolyrange)
    
elif feature_scaling == 2:
    
    #Feature scaling and mean normalisation
    Xtrainmin=np.min(Xtrain,0)
    Xtrainmax=np.max(Xtrain,0)
    Xtrainrange=Xtrainmax-Xtrainmin
    Xtrainmean=np.mean(Xtrain,0)
    
    Xtrainmean[0]=0
    Xtrainrange[0]=1
    
    Xtrainmn=np.subtract(Xtrain,Xtrainmean)
    Xtrainmn=np.divide(Xtrainmn,Xtrainrange)
    Xvalmn=np.subtract(Xval,Xtrainmean)
    Xvalmn=np.divide(Xvalmn,Xtrainrange)
    Xtestmn=np.subtract(Xtest,Xtrainmean)
    Xtestmn=np.divide(Xtestmn,Xtrainrange)
    
    #Define up to N order polynomial
    Xtrainpolymn=udf.create_Xpoly(Xtrainmn,N,mtrain)
    Xvalpolymn=udf.create_Xpoly(Xvalmn,N,mval)
    Xtestpolymn=udf.create_Xpoly(Xtestmn,N,mtest)

#=============================#
#=== Run linear regression ===#
#=============================#
#Run on training data
print("Starting gradient descent (single value of n only)")

#Gradient descent on 1D data
Xtraintemp=Xtrainpolymn[:,0:n+1]
theta,J,J_history,theta_history=udf.lin_reg(Xtraintemp,ytrain,mtrain,k,alpha[n-1],\
                                          number_of_iterations,n)

#Plot J
plt.figure(4)
fig4=plt.semilogy(J_history)

#Plot h
plt.figure(5)
fig5=plt.scatter(Xtrainpolymn[:,1],ytrain)

npoints=100
xplot=np.ones([npoints,N+1])
xplot[:,1]=np.linspace(np.min(Xtrainpolymn[:,1]),np.max(Xtrainpolymn[:,1])\
                            ,npoints)
for i in range(2,N+1):
    xplot[:,i]=np.power(xplot[:,1],i)

yplot=np.dot(theta,xplot[:,0:n+1].T)

fig5=plt.plot(xplot[:,1],yplot)

#==========================================================#
#=== Run linear regression for higher order polynomials ===#
#==========================================================#
print("Starting linear regression for higher order polynomials")

#Initialise variables
theta_history_poly=np.zeros([N,N+1])
J_history_poly=np.zeros([N])

#Train theta on training set
for n in range(1,N+1):
    print("n = ",n)
    
    #Initialise X
    Xtraintemp=Xtrainpolymn[:,0:n+1]
    Xvaltemp=Xvalpolymn[:,0:n+1]
    
    #Train theta on training set
    theta,J,J_history,theta_history=udf.lin_reg(Xtraintemp,ytrain,mtrain,\
                                                k,alpha[n-1],\
                                                number_of_iterations,n)
    
    #Save value of theta
    theta_history_poly[n-1,0:n+1]=theta
    
    #Test different polynomial hypotheses on cross validation set
    J_history_poly[n-1]=udf.compute_cost_unreg(Xvaltemp,yval,mval,theta)
    
#=======================#
#=== Learning curves ===#
#=======================#
#Note: This is carried out for the linear (i.e. non polynomial version) only
print("Starting learning curves")

#Initialise variables
J_learning=np.zeros([mtrain,2])

for mtraintemp in range(mtrain):
    Xtraintemp=Xtrainpolymn[0:mtraintemp,:]
    ytraintemp=ytrain[0:mtraintemp]
    theta,J,J_history,theta_history=udf.lin_reg(Xtraintemp,ytraintemp,\
                                                mtraintemp+1,k,alpha[n-1],\
                                                number_of_iterations,n)
    J_learning_train=udf.compute_cost_unreg(Xtraintemp,ytraintemp,mtraintemp+1,\
                                            theta)
    Xvaltemp=Xvalpolymn
    yvaltemp=yval
    J_learning_val=udf.compute_cost_unreg(Xvaltemp,yvaltemp,mtraintemp+1,theta)
    J_learning[mtraintemp,:]=[J_learning_train,J_learning_val]
    
plt.figure(6)
fig5=plt.semilogy(J_learning[:,0])
fig5=plt.semilogy(J_learning[:,1])
