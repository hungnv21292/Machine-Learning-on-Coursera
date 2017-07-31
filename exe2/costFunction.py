"""
import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    hypothesis = sigmoid(X * theta.T)

    first = np.multiply(-y, np.log(hypothesis))
    second = np.multiply((1 - y), np.log(1 - hypothesis))

    return np.sum(first - second) / (len(X))
"""

def costFunction(theta, X, y, return_grad=False):
# COSTFUNCTION Compute cost and gradient for logistic regression
#   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
#   parameter for logistic regression and the gradient of the cost
#   w.r.t. to the parameters.

    import numpy as np
    from sigmoid import sigmoid

    # Initialize some useful values
    m = len(y)  # Number of traning examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#
    hypothesis = sigmoid(np.dot(X, theta))

    first = y * np.transpose(np.log(hypothesis))
    second = (1 - y) * np.transpose(np.log(1 - hypothesis))

    J = -(1.0/m) * (first + second).sum()

    grad = (1.0/m) * np.dot(hypothesis.T - y, X).T

    if return_grad == True:
        return J, np.transpose(grad)
    elif return_grad == False:
        return J