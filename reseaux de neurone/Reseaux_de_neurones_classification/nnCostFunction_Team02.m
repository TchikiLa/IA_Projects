function [J grad] = nnCostFunction_Team02(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


% Remodelez nn_params dans les paramètres Theta1 et Theta2, les matrices de poids
% pour notre réseau de neurones à 2 couches
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
%fprintf('%d\n',size(Theta1));
%fprintf('%d\n',size(Theta2));

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
%Feedforward le réseau de neurones et renvoyer le coût dans la variable J.

K = num_labels;
X = [ones(m,1) X];

for i = 1:m
	X_i = X(i,:);
	h_of_Xi = sigmoid_Team02( [1 sigmoid_Team02(X_i * Theta1')] * Theta2' );
	

	y_i = zeros(1,K);
	y_i(X(y(i))) = 1; 
	J = J + sum( y_i .* log(h_of_Xi) + (1 - y_i) .* log(1 - h_of_Xi));
end;

J = (-1 / m) * J;
Theta1s=Theta1(:,2:end);
Theta2s=Theta2(:,2:end);
% la regularisation

J = J + (lambda / (2 * m) * (sum(sum(Theta1s.^2)) + sum(sum(Theta2s.^2)))); 

delta_accum_1 = zeros(size(Theta1));
delta_accum_2 = zeros(size(Theta2));

for t = 1:m
	a1 = X(t,:);  
	z2 = a1 * Theta1';
	a2 = [1 sigmoid_Team02(z2)];
	z3 = a2 * Theta2';
	a3 = sigmoid_Team02(z3);
	yi = zeros(1,K);
	yi(X(y(t))) = 1;
	
	delta_3 = a3 - yi;
	delta_2 = delta_3 * Theta2 .* sigmoidGradient_Team02([1 z2]);
	
	delta_accum_1 = delta_accum_1 + delta_2(2:end)' * a1;
	delta_accum_2 = delta_accum_2 + delta_3' * a2;
end;

Theta1_grad = delta_accum_1 / m;
Theta2_grad = delta_accum_2 / m;


% Implement regularization avec cost function et gradients.


Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1) + lambda / m * Theta1(:, 2:input_layer_size+1);
Theta2_grad(:, 2:hidden_layer_size+1) = Theta2_grad(:, 2:hidden_layer_size+1) + lambda / m * Theta2(:, 2:hidden_layer_size+1);


grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
