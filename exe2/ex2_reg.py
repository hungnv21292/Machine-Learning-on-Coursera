"""
%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
"""

import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import use
from scipy.optimize import fmin_bfgs

use('TkAgg')

from plotData import plotData
from ml import plotData1
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

### ===================== Part 0: Visualization data ================================ ####
### Load data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = np.loadtxt('ex2data2.txt', delimiter=",")
X = data[:, :2]
y = data[:, 2]

plt1, p1, p2 = plotData(X, y)

# Label and Legend
plt1.xlabel('Microchip Test 1')
plt1.ylabel('Microchip Test 2')
plt1.legend((p1, p2), ('y = 1', 'y = 0'), numpoints = 1, handlelength = 0)

plt.show()

raw_input('\nProgram paused. Press enter to continue.\n')


### ===================== Part 1: Regularized Logistic Regression ==================== ###
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled

X = mapFeature(X[:, 0], X[:, 1])
m, n = X.shape

# Initialize fitting parameters
init_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1
lambda_reg = 1

# Compute and display initial cost and gradient for regularized logistic regression
cost = costFunctionReg(init_theta, X, y, lambda_reg)

print('Cost at initial theta (zeros): {:f}'.format(cost))

raw_input('\nProgram paused. Press enter to continue.\n')

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?

# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1 (you should vary this)
lambda_reg = 1

#  Run fmin_bfgs to obtain the optimal theta
#  This function returns theta and the cost
myargs = (X, y, lambda_reg)
theta_opt = fmin_bfgs(costFunctionReg, x0=initial_theta, args=myargs)

# plot Boundary
plotDecisionBoundary(theta_opt, X, y)

# # Labels, title and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('lambda = {:f}'.format(lambda_reg))
plt.show()

# Compute Accuracy on our training set
p = predict(theta_opt, X)

print ('Train Accuracy: {:f}'.format(np.mean(p == y) * 100))

raw_input('\nProgram paused. Press enter to continue.\n')