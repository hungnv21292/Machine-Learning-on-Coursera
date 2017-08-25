function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Add ones to the  X data matrix (Add bias for input layer).
X = [ones(m, 1) X];

% Convert y from (1-10) class into num_lables vectorize
y_vec = eye(num_labels);
y1 = y_vec(y, :);

%%% Map from Layer 1 to Layer 2 %%%
a1 = X;
% Converts to matrix of 5000 examples x (25 units in hidden layer + 1)
z2 = a1*Theta1';
% Compute hypothesis for hidden layer
a2 = sigmoid(z2);


%%% Map from Layer 2 to Layer 3 %%%
% Add bias for hidden layer
a2 = [ones(m, 1) a2];
% Converts to matrix of 5000 examples x numbers of lables.
z3 = a2*Theta2';
% Compute hypothesis for output layer.
a3 = sigmoid(z3);


%%% Compute cost function. %%%%
logisf = (y1).*log(a3) + (1-y1).*log(1-a3);
%J = ((-1/m).*sum(sum(logisf)));  % sum of numbers of example in training set and numbers of lables in ouput layer.

# Regularized cost
Theta1_reg = Theta1(:, 2:end);
Theta2_reg = Theta2(:, 2:end);

J = ((-1/m).*sum(sum(logisf))) + (lambda/(2*m)) .* (sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)));

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

K = num_labels;

% Training set: {(x(1), y(1)), ..., (x(m), y(m))}
% Step 1: Set delta = 0
delta_accum_1 = zeros(size(Theta1));
delta_accum_2 = zeros(size(Theta2));

% Step 2: for-loop over the training examples
for t = 1:m
  % Set a(1) = x(i)
  a_1 = X(t, :);
  % Perform forward propagation to compute a(l) for l = 2,3 (have 3 layer)
  z_2 = a_1 * Theta1';
  a_2 = [1 sigmoid(z_2)];
  
  z_3 = a_2 * Theta2';
  a_3 = sigmoid(z_3);
  
  y_i = zeros(1, K);
  y_i(y(t)) = 1;
  
  % Using y(i), compute sigma(3) = a(3) - y(i)
  sigma3 = a_3 - y_i; % matrix 1x10
  % Compute sigma(2), no have sigma(1)
  sigma2 = sigma3 * Theta2 .* sigmoidGradient([1 z_2]); % matrix 1x26
  
  % update delta
  delta_accum_1 = delta_accum_1 + sigma2(2:end)' * a_1; % matrix a_1 is 1x401
  delta_accum_2 = delta_accum_2 + sigma3' * a_2; % matrix a_2 is 1x26
end;

Theta1_grad = delta_accum_1 / m;
Theta2_grad = delta_accum_2 / m;


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1) + lambda / m * Theta1(:, 2:input_layer_size+1);
Theta2_grad(:, 2:hidden_layer_size+1) = Theta2_grad(:, 2:hidden_layer_size+1) + lambda / m * Theta2(:, 2:hidden_layer_size+1);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
