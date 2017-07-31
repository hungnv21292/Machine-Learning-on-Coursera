def costFunctionReg(theta, X, y, lambda_reg):
# COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
#   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
#   theta as the parameter for regularized logistic regression and the
#   gradient of the cost w.r.t. to the parameters.

    import numpy as np
    from sigmoid import sigmoid

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
    hypothesis = sigmoid(np.dot(X, theta))

    first = y * np.transpose(np.log(hypothesis))
    second = (1-y) * np.transpose(np.log(1 - hypothesis))
    reg_params = (float(lambda_reg) / (2*m)) * np.power(theta[1:theta.shape[0]], 2).sum()

    J = -(1.0/m) * (first + second).sum() + reg_params

    # Applies to j = 1, 2, ..., n - not to j = 0
    grad = (1.0/m) * np.dot(hypothesis.T - y, X).T + (float(lambda_reg) / m)*theta

    # the case of j = 0 (recall that grad is a n+1 vector)
    # since we already have the whole vectorized version, we use that
    grad_no_regularization = (1.0/m) * np.dot(hypothesis.T - y, X).T

    # and then assign only the first element of grad_no_regularization to grad
    grad[0] = grad_no_regularization[0]

    return J