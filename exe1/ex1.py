"""
%%  Machine Learning Online Class - Exercise 1: Linear Regression
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
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
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from wamUpExercise import wamUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

### ================ Part 1: Basic Function =================== ###
# Complete warmUpExercise
from gradientDescentMulti import gradientDescentMulti
from computeCostMulti import computeCostMulti
from featureNormalize import featureNormalize

print('Running warmUpExercise ...\n')
print('5x5 Identity Matrix: \n')

wamUpExercise()

raw_input('Program paused. Press enter to continue.\n')

### ================ Part 2: Ploting ========================== ###
print('Plotting Data ...\n')

data = np.loadtxt('ex1data1.txt', delimiter=",")

X = data[:, 0]  # profit
y = data[:, 1]  # population

X_1 = np.power(X, 2)

X_poly = np.column_stack((X, X_1))

print X

print 'X_poly'
print X_poly

m = len(y)  # Number of training examples

# Plot Data
#plotData(X, y)
#plt.show()

plotData(X, y)
#plt.show()

raw_input('Program paused. Press enter to continue.\n')

### ================ Part 3: Cost and Gradient descent ========= ###
print ('Running Gradient Descent...')

### Method 1: Add a column to X
#A = np.ones((m), dtype=int)
#X = np.c_[A, X] # Add a column of ones to x ( Can use np.column_stack to add a column to X)

### Method 2
#X = np.column_stack((np.ones((m, 1)), X))

# Feature scaling
X_norm, mu, sigma = featureNormalize(X_poly)

X_ones = np.ones((m, 1))

X_padded = np.column_stack((X_ones, X_norm))

X_test = np.column_stack((X_ones, X_poly))

print X_padded

# Initialize fitting parameters
#theta = np.zeros(shape=(2, 1))
theta = np.zeros(shape=(3, 1))

# Some gradient descent settings
iterations = 30000
alpha = 0.05

print('Testing the cost function ...\n')
# Compute and display initial cost

#J = computeCost(X, y, theta)
#J = computeCost(X_padded, y, theta)
J = computeCostMulti(X_padded, y, theta)

print('With theta = [0; 0]\nCost computed = %f\n', J)
print('Expected cost value (approx) 32.07\n')

# Further testing of the cost function
#J = computeCost(X_padded, y, np.array([[-1], [2], [3]]))
#print('\nWith theta = [-1; 2]\nCost computed = %f\n', J)
#print('Expected cost value (approx) 54.24\n')

print('\nRunning Gradient Descent ...\n')
# Run gradient descent
#theta, J_history  = gradientDescent(X, y, theta, alpha, iterations)
#theta, J_history  = gradientDescent(X_padded, y, theta, alpha, iterations)
theta, J_history  = gradientDescentMulti(X_padded, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print theta
print 'Expected theta values (approx)\n'
print '-3.6303\n 1.1664\n\n'

print X_padded[:, 1]

# Plot the linear fit
#result = X.dot(theta).flatten()
#result = X_padded.dot(theta).flatten()
result = X_padded.dot(theta).flatten()
plt.figure(1)
plt.plot(data[:, 0], result, '-', label = "Linear regression")
plt.legend(loc = 4) # 4 is lower right
plt.draw()
plt.show()
#plt.hold(False)

"""

# Predict values for population sizes of 35,000 and 70,000
predict1 = (np.array([1, 3.5])).dot(theta)
print 'For population = 35,000, we predict a profit of:\n'
print predict1*10000

predict2 = (np.array([1, 7])).dot(theta)
print 'For population = 70,000, we predict a profit of:\n'
print predict2*10000

raw_input('Program paused. Press enter to continue.\n')

### ===================== Part 4: Visualizing J(theta_0, theta_1) ============== ###
print ('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# Initialize J_vals to a matrix of 0's
J_vals = np.zeros(shape=(theta0_vals.size, theta1_vals.size))

# Fill out J_vals
for i in xrange(len(theta0_vals)):
    for j in xrange(len(theta1_vals)):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i, j] = computeCost(X, y, t)

print J_vals

# Contour plot
# Because of the way meshgrids work in the surf command, we need to transpose J_vals
# before calling surf, or else the axes will be flipped
J_vals = np.transpose(J_vals)

# Plot J_vals as 20 contours spaced logarithmically between 0.01 and 100
fig = plt.figure(2)
cset = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(1, 3, 20))
fig.colorbar(cset)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.scatter(theta[0][0], theta[1][0], marker= 'x', color='red')
plt.title('Contour, showing minimum')
plt.show(block = False)

# Plot surface
fig = plt.figure(3)
ax = Axes3D(fig)
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm)
fig.colorbar(surf, shrink = 0.5, aspect = 5)

plt.title('Surface')
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show(block = False)

raw_input('Program paused. Press enter to continue.\n')

"""