% (C) Tchikou Lakhdar G03---------Debut----------------------------
function [J, grad] = NNcostgrad_Team9(nn_params, input_layer_size, hidden_layer_size, ...
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

% ========1-Implementation de forward propagation=================================================================
%ajouter le (bias) a la premiere colone de x 
x = [ones(m, 1), x];                 
                        % ======= LAYER 2 =======
z2 = x*Theta1'; % sigmoid input: z = x*thetha1
a2 = NNsigmoid_Team9(z2); %activation unit: a2=g(z)= sigmoid(z) 
a2 = [ones(m, 1), a2]; %ajouter le (bias) a la premiere colone de a2 
                         % ======= LAYER 3 =======
z3 = a2*Theta2';
a3 = NNsigmoid_Team9(z3);

% cree une matrice des classes de taille m*num_labels
%avec la fonction de creation de matrice eye qui rempli le diagonale avec 1
%et 0 partout pui rempliser y_k par les valers de y 
tmp = eye(num_labels);
Y_k  = (tmp(y,:));

% ========2-calculer le cout J(theta) regulariser=================================================================

J = -(sum(sum((Y_k .* log(a3) + (1 - Y_k) .* log(1 - a3)), 2)))/m;% calcule du cout 
% calcule de la regularisation
regularisation = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + regularisation;% resultat du cout regulariser 

% ========3-Implementation de backpropagation pour calculer le partial derivatives==============================

% Initialisation de delta1 et delta2 
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for i = 1:m
    xi = x(i, :);
    yi = Y_k(i, :);
% Forward propagation pour (xi, yi)
% ======= LAYER 2 =======
    z2 = xi*Theta1';
    a2_t = NNsigmoid_Team9(z2);
    % ajouter (bias) au vecteur a2_t
    a2_t = [1, a2_t];
% ======= LAYER 3 =======
    z3 = a2_t*Theta2';
    a3_t = NNsigmoid_Team9(z3);
    
    delta_3 = (a3_t - yi);% calculer output layer error delta(L) = a(L) - y(t)
   
    tmp = Theta2'*delta_3'; % calculer hidden layer error
    % delta(2) ne doit pas prendre en compte le bias
    delta_2 = tmp(2:end, :)' .* NNsigmoidGradient_Team9(z2); % delta(2) = theta2'*deltat(L).*g'(z(2))
    % matrice d'erreur de trsition delta1 et deltat2
    delta1 = delta1 + delta_2'*xi; %deltat1 = deltat1 + a1*delta2
    delta2 = delta2 + delta_3'*a2_t; %deltat2 = deltat2 + a2*deltat3
end


Theta1_grad = (1/m)*delta1 + (lambda/m)*[zeros(size(Theta1, 1), 1), Theta1(:,2:end)];
Theta2_grad = (1/m)*delta2 + (lambda/m)*[zeros(size(Theta2, 1), 1), Theta2(:,2:end)];
% retourner le gradient (theta1 et theta2)
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
% (C) Tchikou Lakhdar G03---------Fin----------------------------