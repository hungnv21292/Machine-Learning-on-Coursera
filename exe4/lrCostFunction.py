"""
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
"""
import numpy as np
import sys


from sigmoid import sigmoid


def lrCostFunction(theta, X, y, lambda_reg, return_grad = False):
    # Initialize some useful values
    m = len(y)

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta
    %
    % Hint: The computation of the cost function and gradients can be
    %       efficiently vectorized. For example, consider the computation
    %
    %           sigmoid(X * theta)
    %
    %       Each row of the resulting matrix will contain the value of the
    %       prediction for that example. You can make use of this to vectorize
    %       the cost function and gradient computations. 
    %
    % Hint: When computing the gradient of the regularized cost function, 
    %       there're many possible vectorized solutions, but one solution
    %       looks like:
    %           grad = (unregularized gradient for logistic regression)
    %           temp = theta; 
    %           temp(1) = 0;   % because we don't add anything for j = 0  
    %           grad = grad + YOUR_CODE_HERE (using the temp variable)
    %
    """

    z = np.dot(X, theta)

    hypothesis = sigmoid(z)

    first = y * np.transpose(np.log(hypothesis))
    second = (1 - y) * np.transpose(np.log(1 - hypothesis))
    reg_params = (float(lambda_reg) / (2 * m)) * np.power(theta[1:theta.shape[0]], 2).sum()

    J = -(1.0/m) * (first + second).sum() + reg_params

    # Applies to j = 1, 2, ..., n - not to j = 0
    grad = (1.0/m) * np.transpose(np.dot(hypothesis.T - y, X)) + (float(lambda_reg) / m)*theta

    # the case of j = 0 (recall that grad is a n+1 vector)
    # since we already have the whole vectorized version, we use that
    grad_no_regularization = (1.0/m) * np.dot(hypothesis.T - y, X).T

    # and then assign only the first element of grad_no_regularization to grad
    grad[0] = grad_no_regularization[0]

    if return_grad:
        return J, grad.flatten()
    else:
        return J


"""
def lrCostFunctionReg(theta, reg, X, y):
    m = y.size

    # hypothesis
    z = np.dot(X, theta)
    h = sigmoid(z)

    J = -1*(1/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))

    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])


def lrgradientReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))

    grad = (1 / m) * X.T.dot(h - y) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]

    return (grad.flatten())
"""