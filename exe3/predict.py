"""
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
"""
import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):

    # Turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1, X.shape[0]))

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    #

    ### Map from Layer 1 to Layer 2
    # Add column of ones as bias unit from input layer to second layer
    X = np.column_stack((np.ones((m, 1)), X))

    a1 = X

    # Convert to matrix of 5000 examples x 26 thetas
    z2 = np.dot(X, np.transpose(Theta1))
    # Sigmoid function converts to p between 0 to 1
    a2 = sigmoid(z2)

    ### Map from Layer 2 to Layer 3
    # add column of ones as bias unit from second layer to third layer
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))

    # Convert to matrix of 5000 examples x num_labels
    z3 = np.dot(a2, np.transpose(Theta2))
    # Sigmoid function converts to p between 0 to 1
    a3 = sigmoid(z3)

    # Get indices as in predictOneVsAll
    p = np.argmax(a3, axis=1)

    # =========================================================================

    return p + 1 # offsets python's zero notation