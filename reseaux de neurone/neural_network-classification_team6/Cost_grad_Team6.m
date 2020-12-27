% (C) Tchikou Lakhdar G03---------Debut----------------------------
function [J, grad] = Cost_grad_Team6(nn_params, input_layer_size, hidden_layer_size, ...
                                   num_labels, x, y, lambda)
%retourner un vecteur de  dérivées partielles apartir du vecteur nn_parametre qui contien les
%les valeurs initiale de theta1 et theta2 avec la fonction predifinis reshape

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(x, 1);      
% Initilisation des variables a retourner
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%% =========== 1-Forward propagation=================================================================
x = [ones(m, 1), x]; 
a3 = forwardPropagation_Team6( x,m,Theta1,Theta2 );
                            % cree une matrice de classes de taille m*num_labels
                            %avec la fonction de creation de matrice eye qui rempli le diagonale avec 1
                            %et 0 partout pui rempliser y_k par les valers de y 
tmp = eye(num_labels);
Y_k  = (tmp(y,:));

%% =========== 2-calculer le cout J(theta) regulariser=================================================================

J = -(sum(sum((Y_k .* log(a3) + (1 - Y_k) .* log(1 - a3)), 2)))/m;% calcule du cout 
% calcule de la regularisation
regularisation = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + regularisation;% resultat du cout regulariser 

%% =========== Backpropagation pour calculer le partial derivatives==============================
[ delta1,delta2 ] = backproPagation_Team6(  x,Y_k,m,Theta1,Theta2 );

%% ===========                                =============

Theta1_grad = (1/m)*delta1 + (lambda/m)*[zeros(size(Theta1, 1), 1), Theta1(:,2:end)];
Theta2_grad = (1/m)*delta2 + (lambda/m)*[zeros(size(Theta2, 1), 1), Theta2(:,2:end)];
% retourner le gradient (theta1 et theta2)
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
% (C) Tchikou Lakhdar G03---------Fin----------------------------