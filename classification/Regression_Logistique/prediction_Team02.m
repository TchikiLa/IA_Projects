function [theta, prediction,vect_cost] = prediction_Team02(X,Y,Xtest,labels,alpha,iter,lambda)
  %Taille de tuples
  m = size(X, 1);
  %Nombre de colonnes 
  n = size(X, 2);
  % Initialisation d'une matrice qui contient les valeurs des théta pour
  % chaque classe , donc elle contient un nombre de lignes est égale à
  % nombre de labels et un nombre d'attributs de nos données
  all_theta = zeros(labels, n + 1);
  %Initialisation du vecteur qui contiendra les predictions
  prediction = zeros(size(X, 1), 1);
  % Ajout des uns a la matrice de données
  X = [ones(m, 1) X];
  %Initialisation de la matrice theta
  theta_initial = zeros(n + 1, 1);
  vecteur = zeros(iter,labels);
  for c = 1 :labels
    %Calcul du theta pour chaque classe c en la mettant à 1 et les autres classes à 0
	  [theta,vecteur(:,c)] = gradientDescent_Team02(X, (Y == c), theta_initial, alpha, iter,lambda);
	  % L'enregistrement des thétas de chaque classe dans la matrice all theta
    all_theta(c,:) = theta';
  end
  vect_cost = min(vecteur');
  vect_cost(:);
  %Ajout des uns a la matrice des tests
  Xtest = [ones(size(Xtest, 1), 1) Xtest];
  %Calcul des probabilités de chaque classe, l'indice de la ligne ayant l'hypothese la plus elevé sera la prediction de la classe
  [probability indices] = max(hypothese_Team02(all_theta * Xtest'));
  prediction = indices';
end