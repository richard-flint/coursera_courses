# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:00:51 2020

@author: richa
"""

# Import packages
import numpy as np

#Define functions
def compute_cost(X_transpose,y,theta,m):
    h=np.dot(theta,X_transpose)
    J=(1/(2*m))*sum(((h-y)*(h-y)))
    return J