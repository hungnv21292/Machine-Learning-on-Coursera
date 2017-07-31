"""
Machine Learning Online Class
Exercise 1: Linear regression with multiple variables

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).

"""
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from wamUpExercise import wamUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

from computeCostMulti import computeCostMulti

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn

### ================ Part 1: Feature Normalization ================== ###
print 'Loading data ...\n'

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=",")

X = data[:, :2]
y = data[:, 2]

m = len(y) # number of training examples

print ('First 10 examples from the dataset: \n')
for i in xrange(10):
    print ("x = [{:.0f} {:.0f}], y = {:.0f}".format(X[i, 0], X[i, 1], y[i]))

# Scale features and set them to zero mean
print 'Normalizing Features ...\n'

X_norm, mu, sigma = featureNormalize(X)

# Add intercept term to X (add a column of ones to x)
X_padded = np.column_stack((np.ones((m, 1)), X_norm))

raw_input('Program paused. Press enter to continue.\n')

## ================ Part 2: Gradient Descent ================

print ('Running gradient descent ...')

# perform linear regression on the data set
alpha1 = 0.001
alpha2 = 0.01
alpha3 = 0.05
num_iters = 400

# Initialize Theta and run Gradient Descent
theta = np.zeros((3, 1))

theta_1 = np.zeros((2, 1))

theta1, J_history1 = gradientDescentMulti(X_padded, y, theta, alpha1, num_iters)
theta2, J_history2 = gradientDescentMulti(X_padded, y, theta, alpha2, num_iters)
theta3, J_history3 = gradientDescentMulti(X_padded, y, theta, alpha3, num_iters)

theta4, J_history4 = gradientDescentMulti(X_norm, y, theta_1, alpha2, num_iters)

theta5, J_history5 = gradientDescentMulti(X, y, theta_1, alpha2, num_iters)

print 'Theta 5 is:'
print theta5

#get the cost (error) of the model
#computeCostMulti(X, y, theta)

"""
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arrange(num_iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
"""

"""
# Plot the convergence graph
plt.figure(1)
plt.plot(J_history1, '-r', label='Learning rate: 0.001')
plt.plot(J_history2, '-b', label='Learning rate: 0.01')
plt.plot(J_history3, '-k', label='Learning rate: 0.05')

plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Error vs. Training Epoch')
plt.legend(loc = 1) # 4 is up right
plt.show()
"""

# Display gradient descent's result
print ('Theta computed from gradient descent: ')
print ("{:f}, {:f}, {:f}".format(theta2[0, 0], theta2[1, 0], theta[2, 0]))

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
area_norm = (1650 - float(mu[:, 0])) / float(sigma[:, 0])
br_norm = (3 - float(mu[:, 1])) / float(sigma[:, 1])
house_norm_padded = np.array([1, area_norm, br_norm])

price = np.array(house_norm_padded).dot(theta2)

# ============================================================

print("Predicted price of a 1650 sq-ft, 3 br house using gradient descent:\n ${:,.2f}".format(price[0]))

raw_input('\nProgram paused. Press enter to continue.\n')


# ================== Part 3: Normal Equations ========================= ####
print ('Solving with normal equations...\n')

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#

## Load data
data = np.loadtxt('ex1data2.txt', delimiter=",")
X = data[:, :2]
y = data[:, 2]

# Number of examples
m = len(y)

# Add intercept term to X
X_padded = np.concatenate((np.ones((m, 1)), X), axis=1)
# X_padded = np.column_stack((np.ones((m, 1)), X))

# Calculate the parameters from the normal eqution
theta = normalEqn(X_padded, y)

# Display normal equation's result
print ('Theta computed from the normal equations:\n')
print '%s \n' % theta

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1, 1650, 3]).dot(theta)

print ('Predicted price of a 1650 sq-ft, 3 br house \n')
print '(using normal equations):\n $%f\n' % price

### ======== Part 4: Visualazation ============== ###

### ===================== Part 4: Visualizing J(theta_0, theta_1) ============== ###
print ('Visualizing J(theta_0, theta_1) ...\n')

"""
# Grid over which we will calculate J
theta0_vals = np.linspace(90000, 110000, 1000)
theta1_vals = np.linspace(2000, 4000, 100)

# Initialize J_vals to a matrix of 0's
J_vals = np.zeros(shape=(theta0_vals.size, theta1_vals.size))

# Fill out J_vals
for i in xrange(len(theta0_vals)):
    for j in xrange(len(theta1_vals)):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i, j] = computeCost(X_norm, y, t)

# Contour plot
# Because of the way meshgrids work in the surf command, we need to transpose J_vals
# before calling surf, or else the axes will be flipped
J_vals = np.transpose(J_vals)

# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
fig = plt.figure(1)
cset = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(0, 10, 2000))
#fig.colorbar(cset)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.scatter(theta4[0][0], theta4[1][0], marker= 'x', color='red')
plt.title('Contour, showing minimum')
plt.show()
"""

# Grid over which we will calculate J
theta0_vals = np.linspace(-10000000, 10000000, 100)
theta1_vals = np.linspace(-10000000, 10000000, 100)

# Initialize J_vals to a matrix of 0's
J_vals = np.zeros(shape=(theta0_vals.size, theta1_vals.size))

# Fill out J_vals
for i in xrange(len(theta0_vals)):
    for j in xrange(len(theta1_vals)):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i, j] = computeCostMulti(X, y, t)

print J_vals

# Contour plot
# Because of the way meshgrids work in the surf command, we need to transpose J_vals
# before calling surf, or else the axes will be flipped
J_vals = np.transpose(J_vals)

# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
fig = plt.figure(1)
cset = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 20, 50))
#fig.colorbar(cset)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.scatter(theta4[0][0], theta4[1][0], marker= 'x', color='red')
plt.title('Contour, showing minimum')
plt.show()