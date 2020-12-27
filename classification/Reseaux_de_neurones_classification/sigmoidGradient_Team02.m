function g = sigmoidGradient(z)
%SIGMOIDGRADIENT retourne le gradient de le fonction sigmoid (z)
g = zeros(size(z));
g=sigmoid_Team02(z).*(1-sigmoid_Team02(z));

end
