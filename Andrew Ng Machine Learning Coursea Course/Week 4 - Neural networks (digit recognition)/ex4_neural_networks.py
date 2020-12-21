# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:02:06 2020

@author: richard_dev

Summary: This script implements a neural network algorithm to identify the
digits 0 to 9 in greyscale images. Note that This implementation estimates
its own value for theta using forward and backward propagation. This uses the 
same dataset as Week 3. 
"""

#Clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import ex4_neural_networks_functions as udf

# ========================================================================== #
# =========================== Define model parameters ====================== #
# ========================================================================== #
regularisation = 1          #0=OFF, 1=ON
random_initialisation = 0   #0=OFF, 1=ON This currently doesn't work when on!
is_gradient_check=1         #0=OFF, 1=ON
k=1
epsilon_theta_initialise=0.12
epsilon_grad_check=1E-4
number_of_iterations=10
alpha=0.01

# ========================================================================== #
# ================================ Get data ================================ #
# ========================================================================== #

#Get data
ex4data1=loadmat('ex4data1.mat')
ex4weights=loadmat('ex4weights.mat')

#Unpack data
X=ex4data1['X']
y=ex4data1['y']
theta1_preloaded=ex4weights['Theta1']
theta2_preloaded=ex4weights['Theta2']

#Redefine the 0 value in the y data
y=y[:,0]

#Add x0 column to X
x0=np.ones(np.shape(X)[0])
X=np.column_stack([x0,X])

#Define data characteristics
m=np.shape(X)[0]
n=np.shape(X)[1]

#Plot individual numbers
row_num=2050
x=np.reshape(X[row_num,1:],[20,20])
x1=np.linspace(1,20,20)
x2=np.linspace(1,20,20)
plt.contourf(x1,x2,x,cmap='gray_r')

#Plot y
plt.figure(2)
fig2=plt.plot(y)

#Separate y into vectors for y=1,y=2,y=3 etc.
Y=np.zeros([m,10])
for i in range(10):
    Y[:,i]=udf.create_binary_y_variable(y,i)

# ========================================================================== #
# =========================== Initialise variables ========================= #
# ========================================================================== #
    
#Initialise theta1
theta1=((np.random.rand(25,401)*2)-1)*epsilon_theta_initialise 
if random_initialisation == 0:
    theta1=theta1_preloaded                 #theta1=[25,401]

#Initialise theta2
theta2=((np.random.rand(10,26)*2)-1)*epsilon_theta_initialise
if random_initialisation == 0:
    theta2=theta2_preloaded                 #theta2=[10,26]

#J_history
J_history=np.zeros([number_of_iterations+1])
  
# ========================================================================== #
# =================================== ML code ============================== #
# ========================================================================== # 

for ind in range(number_of_iterations):
    
    # Initialise Delta1 and Delta2
    Delta1=np.zeros([np.shape(theta1)[0],np.shape(theta1)[1]])
    Delta2=np.zeros([np.shape(theta2)[0],np.shape(theta2)[1]])
    
    #Initialise D
    D1=np.zeros([np.shape(theta1)[0],np.shape(theta1)[1]])
    D2=np.zeros([np.shape(theta2)[0],np.shape(theta2)[1]])

    for t in range(m):
        # ==================== #
        # === Feed forward === #
        # ==================== #
        # This computes the hypothesis h from the existing model parameters
        # This also allows us to calculate J
        
        #Compute a1
        a1=np.copy(X[t,:])                                  #a1=[401,]
    
        #Compute a2
        z2=np.dot(theta1,a1)                                #z2=[25,]
        a2=udf.sigmoid_function(z2)                         #a2=[25,]
        a2=np.hstack([[1],a2])                              #a2=[26,]
                                                            #Add bias layer
        
        #Compute a3
        z3=np.dot(theta2,a2)                                #z3=[10,]
        a3=udf.sigmoid_function(z3)                         #a3=[10,]
        
        #Compute h
        h=np.copy(a3)                                       #h=[10,]
        
        # ======================== #
        # === Back propagation === #
        # ======================== #
        # This calculates the gradient dJ/Dtheta from exiting model parameters
        # This can then be used in gradient descent and/or other iterative 
        # algorithms.
        
        #Compute delta3
        delta3=a3-Y[t,:]                            #delta3=[10,]
        
        #Compute delta2
        d21=np.dot(delta3,theta2)                   #d21=[26,]
        d22=np.multiply(a2,1-a2)                    #d22=[26,]
        delta2=np.multiply(d21,d22)                 #delta2=[26,]
        
        #Reshape for dot product
        delta3=np.reshape(delta3,[len(delta3),1])
        delta2=np.reshape(delta2[1:],[len(delta2)-1,1])
        a2=np.reshape(a2,[len(a2),1])
        a1=np.reshape(a1,[len(a1),1])
        
        #Compute Delta2
        Delta2=Delta2+np.dot(delta3,a2.T)             #Delta2=[10,26]
        
        #Compute Delta1
        Delta1=Delta1+np.dot(delta2,a1.T)             #Delta1=[25,401]
        
    #Compute D
    if regularisation == 0:
        D1=(1/m)*Delta1
        D2=(1/m)*Delta2
    elif regularisation == 1:
        #Bias unit
        D1[:,0]=(1/m)*Delta1[:,0]
        D2[:,0]=(1/m)*Delta2[:,0]
        #All other units
        D1[:,1:]=(1/m)*(Delta1[:,1:]+k*theta1[:,1:])
        D2[:,1:]=(1/m)*(Delta2[:,1:]+k*theta2[:,1:])
        
    # ====================== #
    # === Gradient Check === #
    # ====================== #
    
    # Gradient check is used to check the back propagation algorithm.
    # It is quite slow so we only check the first iteration.
    # We should see a difference of around only 1E-9.
    
    if ind == 0 and is_gradient_check==1:
        D1_difference,D2_difference=udf.gradient_check(h,Y,theta1,theta2,regularisation,\
                                          k,m,epsilon_grad_check,D1,D2)
    
    # ======================== #
    # === Gradient Descent === #
    # ======================== #
    
    # Update theta
    theta1=theta1-alpha*D1
    theta2=theta2-alpha*D2
    
    # Calculate cost
    J=udf.compute_cost(h,Y,theta1,theta2,regularisation,k,m)
    
    #Save the cost
    J_history[ind+1]=J

# ========================================================================== # 
# ======================== Accuracy with training data ===================== #
# ========================================================================== # 

#Feed forward to find final hypothesis
h,a1,a2,a3=udf.feed_forward(X,theta1,theta2)

#Estimated numbers
max_per_row=np.amax(abs(h),1)
estimated_numbers=np.zeros(m)
for i in range(m):
    estimated_number=np.where(abs(h[i,:])==max_per_row[i])
    estimated_numbers[i]=estimated_number[0].item()
estimated_numbers=estimated_numbers+1
estimated_numbers[estimated_numbers==10]=0
    
#Calculate percentage of correct answers
correct_numbers=(estimated_numbers==y)*1
percentage_correct=(sum(correct_numbers)/m)*100
print('Percentage correct = ',percentage_correct,'%')