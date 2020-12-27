function p = predict_Team02(Theta1, Theta2, X)
%Pr�dire le libell� d�une entr�e � partir d�un r�seau de neurones form�

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

h1 = sigmoid_Team02([ones(m, 1) X] * Theta1');
h2 = sigmoid_Team02([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);
end
