import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):

    #GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = y.size # number of training examples

    J_history = np.zeros(shape=(num_iters, 1))

    #temp = np.zeros(shape=(2, 1))
    temp = np.zeros(shape=(3, 1))

    print X


    for i in range(num_iters):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #

        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]
        errors_x3 = (predictions - y) * X[:, 2]

        temp[0] = theta[0] - alpha * (1.0 / m)*errors_x1.sum()
        temp[1] = theta[1] - alpha * (1.0 / m)*errors_x2.sum()
        temp[2] = theta[2] - alpha * (1.0 / m)*errors_x3.sum()

        theta = temp

        # ============================================================

        # Save the cost J in every iteration
        #J_history[i, 0] = computeCost(X, y, theta)

    return theta, J_history


