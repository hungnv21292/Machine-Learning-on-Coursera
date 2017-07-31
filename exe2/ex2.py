"""
%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
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

import numpy as np

from plotData import plotData
from sigmoid import sigmoid
from costFunction import costFunction
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

import scipy.optimize as opt
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data1.txt', delimiter=",")
X = data[:, :2]
y = data[:, 2]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plot, p1, p2 = plotData(X, y)

# Add Label and Legend
plot.xlabel('Exam 1 score')
plot.ylabel('Exam 2 score')
plot.legend((p1, p2), ('Admitted', 'Not Admitted'), numpoints=1, handlelength=0)
plot.show()

raw_input('\n Program paused. Press enter to continue.\n')

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

X = np.column_stack((np.ones((m, 1)), X))

# Initialize fitting parameters
init_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
cost, grad = costFunction(init_theta, X, y, return_grad = True)

print ('Cost at initial theta (zeros): {:f}'.format(cost))
print ('Gradient at initial theta (zeros):')
print grad

raw_input('\n Program paused. Press enter to continue.\n')

## ============= Part 3: Optimizing using fmin (and fmin_bfgs)  =============
#  In this exercise, you will use a built-in function (fmin) to find the
#  optimal parameters theta.

#  Run fmin and fmin_bfgs to obtain the optimal theta
#  This function will return theta and the cost

my_args = (X, y)
theta = fmin(costFunction, x0 = init_theta, args= my_args)
theta_opt, cost_at_theta, _, _, _, _, _ = fmin_bfgs(costFunction, x0 = theta, args = my_args, full_output=True)

#theta_opt, nfeval, rc = opt.fmin_tnc(func=costFunction, x0=init_theta, fprime=grad, args=my_args)

# Print theta to screen
print ('Cost at theta found by fmin: {:f}'.format(cost_at_theta))
print ('theta_opt:'),
print theta

# Plot Boundary
plotDecisionBoundary(theta_opt, X, y)
plot.show()

raw_input('\n Program paused. Press enter to continue.\n')

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid(np.dot(np.array([1, 45, 85]), theta_opt))

print('For a student with scores 45 and 85, we predict an admission probability of {:f}'.format(prob))

# Compute accuracy on our training set

p = predict(theta_opt, X)

print('Train Accuracy: {:f}'.format(np.mean(p == y)*100))

raw_input('\n Program paused. Press enter to continue')