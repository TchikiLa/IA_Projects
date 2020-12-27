function [j] = cost_Team02(X, y , theta,lambda) 
  m = size(X,1);
  j = (-1/m) * sum(y.*log(hypothese_Team02(X * theta)) + (1 - y).*log(1 - hypothese_Team02(X * theta))) + (lambda/(2*m)) * sum(theta.^2);
end
