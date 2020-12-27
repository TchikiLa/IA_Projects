% (C) Tchikou Lakhdar G03---------Debut----------------------------
function p = NNpredict_Team9(Theta1, Theta2, X)
%la fonction predict prend en entre les valeurs des parametres (theta) et
%les valeur en entre X pui retoure les valeurs predits sous forme de
%d'index du vecteur du resultat 
% la valeur en retour est entre 1:8 representant les 8 classe difinit dans
% notre probleme de prediction
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);
h1 = NNsigmoid_Team9([ones(m, 1) X] * Theta1');
h2 = NNsigmoid_Team9([ones(m, 1) h1] * Theta2');
[~, p] = max(h2, [], 2);

end
% (C) Tchikou Lakhdar G03---------Fin----------------------------