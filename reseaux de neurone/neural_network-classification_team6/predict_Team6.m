% (C) Tchikou Lakhdar G03---------Debut----------------------------
function p = predict_Team6(Theta1, Theta2, X)
%la fonction predict prend en entre les valeurs de parametres theta et
%les valeur en entrées X pui retoure les valeurs predits sous forme de
%d'index du vecteur du resultat 
% la valeur en retour est entre [1-3] representant les 3 classes difinit 
% dans notre probleme de prediction
s = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);
h1 = sigmoid_Team6([ones(s, 1) X] * Theta1');
h2 = sigmoid_Team6([ones(s, 1) h1] * Theta2');
[~, p] = max(h2, [], 2);
end
% (C) Tchikou Lakhdar G03---------Fin----------------------------