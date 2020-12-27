% (C) Tchikou Lakhdar G03---------Debut----------------------------
function g = NNsigmoid_Team9(z)
%calcule la sigmoid activation, fonction qui retourne la valeur a de avtivation of unit
g = 1.0 ./ (1.0 + exp(-z));%g = SIGMOID(z) calcule le sigmoid de Z
end
% (C) Tchikou Lakhdar G03---------Fin----------------------------