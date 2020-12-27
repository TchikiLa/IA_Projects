function [h]=hypothese_Team02(e)
  % Calcule de l'hypothese 
  h = (1 ./ (1 + exp(-e)));
end
