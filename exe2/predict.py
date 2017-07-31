"""
import numpy as np
from sigmoid import sigmoid

def predict(theta, X):
    probability = sigmoid(X.dot(theta))

    return [1 if x >= 0.5 else 0 for x in probability]
"""

def predict(theta, X):
# PREDICT Predict whether the label is 0 or 1 using learned logistic
# regression parameters theta
#   p = PREDICT(theta, X) computes the predictions for X using a
#   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    import numpy as np
    from sigmoid import sigmoid

    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters.
#               You should set p to a vector of 0's and 1's
#

    sigValue = sigmoid(np.dot(X, theta))
    p = sigValue >= 0.5

    return p