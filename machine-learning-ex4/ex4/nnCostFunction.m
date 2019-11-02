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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_re= zeros(length(y),num_labels);
for i=1:length(y)
  y_re(i,y(i,1))= 1;
end

X = [ones(m, 1) X];
a=sigmoid(X*Theta1');
a= [ones(m,1) a];
b=(sigmoid(a*Theta2'));
J= -sum(diag((y_re'*log(b) + (1-y_re)'*log(1-b))))/m + lambda*sum(diag((Theta1(:,2:size(Theta1,2)))*Theta1(:,2:size(Theta1,2))'))/(2*m)+ lambda*sum(diag((Theta2(:,2:size(Theta2,2)))*Theta2(:,2:size(Theta2,2))'))/(2*m);
%J= -sum(y'*log(b) + (1-y)'*log(1-b))/m + lambda*sum(sum((Theta1(2:size(Theta1,1),:))'*Theta1(2:size(Theta1,1),:)))/(2*m)+ lambda*sum(sum((Theta2(2:size(Theta2,1),:))'*Theta2(2:size(Theta2,1),:)))/(2*m);
Del1 = zeros(size(Theta1));
Del2 = zeros(size(Theta2));

for t=1:m
  z2=X(t,:)*Theta1'; % 1 X 25
  a2=sigmoid(z2); % 1 X 25
  a2= [ones(1,1) a2];
  z3=a(t,:)*Theta2'; % 1 X 10
  a3=sigmoid(z3); % 1 X 10
  del_2= zeros(1,26);
  del_3 = zeros(1,10);
  del_3 = a3 - y_re(t,:); % 1 X 10
  sg= sigmoidGradient(z2);
  sg = [ones(1,1) sg];
  del_2 = (del_3*Theta2).*sg; %1 X 26
  del_2 = del_2(:,2:length(del_2)); % 1 X 25
  Del2 = Del2 + del_3'*a2;
  Del1 = Del1 + del_2'*X(t,:);
end
Theta1_grad(:,1) = Del1(:,1)/m;
Theta2_grad(:,1) = Del2(:,1)/m;
Theta1_grad(:,2:size(Theta1,2)) = (Del1(:,2:size(Theta1,2))+ (lambda*Theta1(:,2:size(Theta1,2))))/m;
Theta2_grad(:,2:size(Theta2,2)) = (Del2(:,2:size(Theta2,2))+ (lambda*Theta2(:,2:size(Theta2,2))))/m;










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
