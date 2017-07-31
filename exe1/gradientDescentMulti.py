import numpy as np
from computeCostMulti import computeCostMulti
from computeCost import computeCost

def gradientDescentMulti(X, y, theta, alpha, num_iters):

    #GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    #   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y)  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for i in xrange(num_iters):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #

        hypothesis = X.dot(theta)

        errors = hypothesis - np.transpose([y])

        theta = theta - alpha * (1./m) * (np.transpose(X).dot(errors))
        # ============================================================

        # Save the cost J in every iteration
        #J_history[i] = computeCost(X, y, theta)

    return theta, J_history