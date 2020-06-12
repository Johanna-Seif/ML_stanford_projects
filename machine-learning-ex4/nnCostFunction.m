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
nb_output = size(Theta2, 1);

% ====================== YOUR CODE HERE ======================

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Adding the bias to X
A1 = [ones(m, 1) X]';
% In the following each column corresponds to one data
Z2 = Theta1 * A1;
A2 = sigmoid(Z2); % Computing the second second layer in vectorial
A2 = [ones(1, m); A2]; % Adding the bias
Z3 = Theta2 * A2;
h_X = sigmoid(Z3); % Computing the last second layer in vectorial
% h_X(:, mu) is one input of the NN

% Transform y into vectors,
  % each column is the expected output for the correspondint input in X'
y_vect = zeros(nb_output, m);
for sample_nb = 1:m
  expected_output = y(sample_nb);
  y_vect(expected_output, sample_nb) = 1;
end

% Compute the cost in vectorial
J = sum ( sum ( (-y_vect) .* log(h_X) - (1-y_vect) .* log(1-h_X) )) / m;

% Regularization of the cost
Theta1_no_bias = Theta1(:, 2:size(Theta1, 2));
Theta2_no_bias = Theta2(:, 2:size(Theta2, 2));
regularized_term = sum ( sum (Theta1_no_bias .^ 2) ) + ...
                  sum ( sum (Theta2_no_bias .^ 2) );
J = J + lambda * regularized_term / (2 * m);

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

% Computation of the error rate for each data
for t = 1:m
  % Extracting info from forward propagation.
  a3 = h_X(:, t);   % Size  10 * 1
  a2 = A2(:, t);    % Size  26 * 1
  a1 = A1(:, t);    % Size 401 * 1
  y = y_vect(:, t); % Size  10 * 1

  % Computing errors
  delta3 = a3 - y;          % Size 10 * 1
  delta2 = Theta2' * delta3 .* a2 .* (1 - a2);
  delta2 = delta2(2:end);   % Size 25 * 1

  % Accumulate gradient
  Theta1_grad += delta2 * a1';
  Theta2_grad += delta3 * a2';

end


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad += lambda * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad += lambda * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
