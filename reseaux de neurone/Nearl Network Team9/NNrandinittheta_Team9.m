% (C) Tchikou Lakhdar G03---------Debut----------------------------
function W = NNrandinittheta_Team9(L_in, L_out)
%cet fonction prend en parametre le nombre d'unités d'un layer en entree et
%en sortie puis retourn une matrice (l_out * l_in+1) de petites valeur random 
W = zeros(L_out, 1 + L_in);
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * (2 * epsilon_init) - epsilon_init;

end
% (C) Tchikou Lakhdar G03---------Fin----------------------------