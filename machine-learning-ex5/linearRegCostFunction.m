function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

% Computing the cost_vector
hx = X * theta;
gap_X_y = hx - y;
cost_vector = (gap_X_y .^ 2);
reg_vector = theta(2:end) .^ 2;
J = (sum(cost_vector) + lambda * sum(reg_vector)) / (2*m);

% Computing the gradient
grad = (X' * (hx - y) + lambda * [0; theta(2:end)]) / m;
% =========================================================================

grad = grad(:);

end
