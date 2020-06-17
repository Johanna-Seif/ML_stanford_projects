function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%
%

results = [];

% Testing result on cross validation set for all elements
for sigma_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
  for C_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    % Choosing the model on which we train the data
    model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    % Compute the prediction given the model on the CV set
    predictions = svmPredict(model, Xval);
    % Compute the error of this prediction
    error_test = mean(double(predictions ~= yval));
    results = [results; [error_test, sigma_test, C_test] ];
  end
end

sorted_results = sortrows(results); % sort matrix with increasing error

sigma = sorted_results(1,2);
C = sorted_results(1,3);

% =========================================================================

end
