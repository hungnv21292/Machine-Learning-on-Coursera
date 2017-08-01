"""
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
"""

import numpy as np
from scipy.optimize import minimize

from lrCostFunction import lrCostFunction
#from lrCostFunction import lrCostFunctionReg
#from lrCostFunction import lrgradientReg


def oneVsAll(X, y, num_lables, lambda_reg):
    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_lables, n+1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: You should complete the following code to train num_labels
    %               logistic regression classifiers with regularization
    %               parameter lambda. 
    %
    % Hint: theta(:) will return a column vector.
    %
    % Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
    %       whether the ground truth is true/false for this class.
    %
    % Note: For this assignment, we recommend using fmincg to optimize the cost
    %       function. It is okay to use a for-loop (for c = 1:num_labels) to
    %       loop over the different classes.
    %
    %       fmincg works similarly to fminunc, but is more efficient when we
    %       are dealing with large number of parameters.
    %
    % Example Code for fmincg:
    %
    %     % Set Initial theta
    %     initial_theta = zeros(n + 1, 1);
    %     
    %     % Set options for fminunc
    %     options = optimset('GradObj', 'on', 'MaxIter', 50);
    % 
    %     % Run fmincg to obtain the optimal theta
    %     % This function will return theta and the cost 
    %     [theta] = ...
    %         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    %                 initial_theta, options);
    %
    """

    for c in xrange(num_lables):
        # Set initial theta
        init_theta = np.zeros((n + 1, 1))

        print("Training {:d} out of {:d} categories...".format(c + 1, num_lables))

        # Optimize

        my_args = (X, (y%10 == c).astype(int), lambda_reg, True)
        theta_opt = minimize(lrCostFunction, x0 = init_theta, args=my_args, options={'disp': True, 'maxiter':13}, method = 'Newton-CG', jac = True)

        # Assign row of all_theta coressponding to current class
        all_theta[c, :] = theta_opt["x"]

        """
        for c in np.arange(1, num_lables + 1):
            theta_opt = minimize(lrCostFunctionReg, init_theta, args=(lambda_reg, X, (y%10 == c) * 1), method=None,
                           jac=lrgradientReg, options={'maxiter': 50})

            all_theta[c - 1] = theta_opt.x
        """
        # =========================================================================

    return all_theta