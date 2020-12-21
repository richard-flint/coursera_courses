# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:39:04 2020

@author: richard_dev

Summary: This script uses a support vector machine (SVM) to classify spam emails.
The script uses a pre-populated training dataset that includes feature vectors
for both spam and non-spam emails.
"""

#Clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

#Imports
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.svm import LinearSVC
import scipy

#===================#
#=== Model setup ===#
#===================#

sigma=np.array([0.01,0.03,0.1,0.3,1,3,10,30,100,300])
C=np.array([0.01,0.03,0.1,0.3,1,3,10,30,100,300])
indc=3
iterate=0
N=10 #For top N words in classifier

#====================#
#=== Prepare data ===#
#====================#

#Get data
spamTrain=loadmat('spamTrain.mat')
spamTest=loadmat('spamTest.mat')
vocab = pd.read_csv(r"vocab.txt", sep='\t')

#Separate into variables
Xtrain=spamTrain['X']
ytrain=spamTrain['y']
Xtest=spamTest['Xtest']
ytest=spamTest['ytest']

#Reformat variables
ytrain=ytrain[:,0]
ytest=ytest[:,0]
vocab=vocab.to_numpy()
vocab[:,0]=vocab[:,0]-1

#===============#
#=== Run SVM ===#
#===============#

if iterate == 0:
    #Run for one value of C
    clf=LinearSVC(C=C[indc])
    clf.fit(Xtrain,ytrain)
    SCM_performance=clf.score(Xtest,ytest)*100
elif iterate == 1:
    #Calculate further model parameters
    n_sigma=len(sigma)
    n_C=len(C)

    #Initialise further model vectors
    SCM_performance=np.zeros([n_C],dtype='float64')
    
    #Iterate over multiple values C
    for indc in range (n_C):
        print(indc)
        #Run SVM
        clf=LinearSVC(C=C[indc])
        clf.fit(Xtrain,ytrain)
                
        #Calculate performance using the validation set
        SCM_performance[indc]=clf.score(Xtest,ytest)*100

#Output performane on test dataset
print("Performance is: ",SCM_performance,"%")
    
#===========================#
#=== Additional analysis ===#
#===========================#

coefficients=clf.coef_[0]
ranks=scipy.stats.rankdata(coefficients, method='min')
top_ranks=np.ones(N)*np.max(ranks)-np.linspace(0,N-1,N)
top_coefficients=np.zeros(N)
top_vocab=[None]*N
for n in range(N):
    top_coefficient=np.where(ranks==top_ranks[n])[0]
    top_coefficient=top_coefficient[0]
    vocab_index=np.where(vocab[:,0]==top_coefficient)
    top_coefficients[n]=top_coefficient
    top_vocab[n]=vocab[vocab_index[0][0],1]