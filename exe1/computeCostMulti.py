import numpy as np

def computeCostMulti(X, y, theta):

    #COMPUTECOST Compute cost for linear regression
    #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = len(y)  # Number of training examples (Can use len(y) to get number of training examples)

    # m = len(y) # Return the size of the first dimension

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    # note that
    #	theta is an (n+1)-dimensional vector
    #	X is an m x (n+1)-dimensional matrix
    #	y is an m-dimensional vector

    hypothesis = X.dot(theta)

    errors = np.power((hypothesis - np.transpose([y])), 2)

    J = (1.0/(2*m)) * errors.sum(axis = 0) # Sum of columns

    return J
