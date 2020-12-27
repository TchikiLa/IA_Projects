% (C) Tchikou Lakhdar G03---------Debut----------------------------
function g = sigmoid_Team6(z)
%calculé la fonction d'activation g(Z) 
%tel que SIGMOID(z)=g(z)=1/(1+e(-z)) et qui retourne
%la valeur d'activation (a) d'une unité
g = 1.0 ./ (1.0 + exp(-z));
end
% (C) Tchikou Lakhdar G03---------Fin----------------------------