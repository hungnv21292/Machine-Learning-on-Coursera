%% Machine Learning Online Class
%  Exercise 5 | Regularized Linear Regression and Bias-Variance
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     linearRegCostFunction.m
%     learningCurve.m
%     validationCurve.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 2;           % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
%load ('ex5data1.mat');

images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

images = images';

sel = randperm(size(images, 1));
sel = sel(1:100)

% We are using display_network from the autoencoder code
%display_network(images(:,1:100)); % Show the first 100 images
%displayData(images1(:,1:100)); % Show the first 100 images
displayData(images(sel, :)); % Show the first 100 images
%disp(labels(1:10));


images_zero = ones(6000, 784) * 2;
images_two = ones(6000, 784) * 2;
j = 1;
k = 1;

% Data which label= 0 or 2
for i = 1:60000
  if labels(i) == 0
    images_zero(j, :) = images(i, :);
    j = j + 1;
  elseif labels(i) == 2
    images_two(k, :) = images(i, :);
    k = k + 1;
  else
    % nothing
  end
end

fprintf('Program paused. Press enter to continue.\n');
pause;

images_zero_train = images_zero(1:5000, :);
images_zero_test = images_zero(5001:5500, :);   %500 sample

images_two_train = images_two(1:5000, :);
images_two_test = images_two(5001:5005, :);    %5 sample



% Choose random data for "0" digit
images_zero = images_zero_train(1:5000, :);
sel_zero = randperm(size(images_zero, 1));
sel_zero = sel_zero(1:5000);
images_zero = images_zero(sel_zero, :);

%sel_0 = sel_zero(1:100)
%displayData(images_zero(sel_0, :)); % Show the first 100 images


% Choose random data for "2" digit
images_two = images_two_train(1:5000, :);
sel_two = randperm(size(images_two, 1));
sel_two = sel_two(1:5000);
images_two = images_two(sel_two, :);


image_zero_add_label = [images_zero ones(size(images_zero, 1), 1)]
image_two_add_label = [images_two ones(size(images_two, 1), 1)*2]

image_zero_test_add_label = [images_zero_test ones(size(images_zero_test, 1), 1)];
image_two_test_add_label = [images_two_test ones(size(images_two_test, 1), 1)*2];


% Merge data into traning set
images_training = [image_zero_add_label; image_two_add_label];
r = randperm(size(images_training, 1));
images_training = images_training(r, :);


% Merge data into test set
images_test = [image_zero_test_add_label; image_two_test_add_label];
r = randperm(size(images_test, 1));
images_test = images_test(r, :);

X = images_training(:, 1:(end-1));
y = images_training(:, end);

X_test = images_test(:, 1:(end-1));
y_test = images_test(:, end);


images_trial = [images_zero; images_two];
r = randperm(size(images_trial, 1));
images_trial = images_trial(r, :);


sel = randperm(size(images_trial, 1));
sel = sel(1:100)

% We are using display_network from the autoencoder code
%display_network(images(:,1:100)); % Show the first 100 images
%displayData(images1(:,1:100)); % Show the first 100 images
displayData(images_trial(sel, :)); % Show the first 100 images
%disp(labels(1:10));

fprintf('Program paused. Press enter to continue.\n');
pause;

% m = Number of examples
m = size(images_training, 1);

% Plot training data
%plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
%xlabel('Change in water level (x)');
%ylabel('Water flowing out of the dam (y)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 4: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Part 8: Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
% lambda = 3;
% checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(initial_nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

% fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         % '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 2);

%  You should also try different values of lambda
lambda = 3;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

% Creat X-test and y-test have skew data.


pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


pred_test = predict(Theta1, Theta2, X_test);

fprintf('\nValue of pred_test:%f\n', pred_test);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);


