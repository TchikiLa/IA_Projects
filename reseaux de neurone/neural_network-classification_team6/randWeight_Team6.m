% (C) Tchikou Lakhdar G03---------Debut----------------------------
function W = randWeight_Team6(L_in, L_out)
%cet fonction prend en parametre le nombre d'unités
%du layer en entrée et en sortie et retourn la matrice 
%des poids de taille (l_out * l_in+1) de petites valeurs random
W = zeros(L_out, 1 + L_in);
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * (2 * epsilon_init) - epsilon_init;
end
% (C) Tchikou Lakhdar G03---------Fin----------------------------