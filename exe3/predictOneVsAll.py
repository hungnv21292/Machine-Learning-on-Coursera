"""
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1).
"""
import numpy as np
from sigmoid import sigmoid

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the follwing variables correctly
    p = np.zeros((m, 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: Complete the following code to make predictions using
    %               your learned logistic regression parameters (one-vs-all).
    %               You should set p to a vector of predictions (from 1 to
    %               num_labels).
    %
    % Hint: This code can be done all vectorized using the max function.
    %       In particular, the max function can also return the index of the 
    %       max element, for more information see 'help max'. If your examples 
    %       are in rows, then, you can use max(A, [], 2) to obtain the max 
    %       for each row.
    %    
    """

    z = np.dot(X, np.transpose(all_theta))

    hypothesis = sigmoid(z)

    p = np.argmax(hypothesis, axis=1)


    # =========================================================================

    return p