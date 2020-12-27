function [theta , vect_cost] = gradientDescent_Team02(X, y, theta, alpha, iteration,lambda)
  m = length(y); 
  vect_cost = zeros(iteration, 1);
  n = length(theta);
  temp = theta; 
  for it = 1 : iteration
    erreur = (hypothese_Team02(X * theta) - y);
    for j=1 : n
        temp(j,1) = sum(erreur.*X(:,j));
    end
    theta = theta - (alpha/m) * temp + (lambda/m) * temp;
    vect_cost(it,1) = cost_Team02(X,y,theta,lambda);
  end
end