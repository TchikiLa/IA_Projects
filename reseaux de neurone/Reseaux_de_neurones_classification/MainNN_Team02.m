clear; close all; clc;
%data=csvread('Phishing.csv'); 
bdd=load('PiiishingData.txt');


X=bdd(:,1:9);
y=bdd(:,10); %tous les lignes de la derniere colonne
%______________________________________________________

%data de neuron
input_layer_size  = 9;  
hidden_layer_size = 7;  %75% 
num_labels = 3;
%_____________________________________________________
m = size(X, 1);

initial_Theta1 = randWeight_Team02(input_layer_size, hidden_layer_size);
initial_Theta2 = randWeight_Team02(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('Calcule de theta1 et theta2. \n');

% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('Unrolled Params. Appuyez sur une touche pour continue.\n');
pause;

lambda = 0,001;

J = nnCostFunction_Team02(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

%fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         %'\n'], J);
     
fprintf('\nNeural Network... \n')


options = optimset('MaxIter', 50);

lambda = 0.001;

costFunction = @(p) nnCostFunction_Team02(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% costFunction est une fonction qui prend un seul argument
[nn_params, cost] = fmincg_Team02(costFunction, initial_nn_params, options);

% Obtenir Theta1 et Theta2 depuis nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
         
fprintf('\nAppuyez sur une touche pour continue.\n');
pause;
            
pred = predict_Team02(Theta1, Theta2, X);

fprintf('\nLA précision de NN: %f\n', mean(double(pred == y)) * 100);
pause;