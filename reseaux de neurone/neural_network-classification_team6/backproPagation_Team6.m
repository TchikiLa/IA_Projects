function [ delta1,delta2 ] = backproPagation_Team6(  x,Y_k,m,Theta1,Theta2 )
%% =========== Implementation de backpropagation pour calculer le partial derivatives==============================
% Initialisation de delta1 et delta2 
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
for i = 1:m
    xi = x(i, :);
    yi = Y_k(i, :);
% Forward propagation pour (xi, yi)
% ======= LAYER 2 =======
    z2 = xi*Theta1';
    a2_t = sigmoid_Team6(z2);
    % ajouter (bias) au vecteur a2_t
    a2_t = [1, a2_t];
% ======= LAYER 3 =======
    z3 = a2_t*Theta2';
    a3_t = sigmoid_Team6(z3);
    
    delta_3 = (a3_t - yi);% calculer output layer error delta(L) = a(L) - y(t)
   
    tmp = Theta2'*delta_3'; % calculer hidden layer error
    % delta(2) ne doit pas prendre en compte le bias
    delta_2 = tmp(2:end, :)' .* sigmoidDerivative_Team6(z2); % delta(2) = theta2'*deltat(L).*g'(z(2))
    % matrice d'erreur de trsition delta1 et deltat2
    delta1 = delta1 + delta_2'*xi; %deltat1 = deltat1 + a1*delta2
    delta2 = delta2 + delta_3'*a2_t; %deltat2 = deltat2 + a2*deltat3
end
end

