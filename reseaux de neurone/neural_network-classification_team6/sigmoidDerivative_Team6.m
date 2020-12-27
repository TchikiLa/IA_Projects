% (C) Tchikou Lakhdar G03---------Debut----------------------------
function g = sigmoidDerivative_Team6(z)
%la fonction sigmoidDerivative calcule le "G-prime 
%ou le derivative terme ou le gradient de la Sigmoid Function

g = zeros(size(z));

%g'(z) = a .* (1-a) |tels que| a = g(z) = sigmoid(z)
g = sigmoid_Team6(z).*(1-sigmoid_Team6(z));

end
% (C) Tchikou Lakhdar G03---------Fin----------------------------