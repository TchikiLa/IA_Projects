% (C) Tchikou Lakhdar G03---------Debut----------------------------
clear ; close all; clc

%% =========== Part 0: Configurer les parametre de notre architecture  =============
input_layer_size  = 4;  % 4 Input variables
hidden_layer_size = 5;   % 5 unitées cachées 
num_labels = 3;          % 3 unitées de sorites, de 1 au 8   

%% =========== Part 1: Chargement des données =============

% Load Training Data
fprintf('\n  Chargement des données ...\n');
data1 = load('train.txt');
data2 = load('test.txt');
data3 = load('validation.txt');
%les donne de traitement
X_train = data1(:, 1:4);
y_train = data1(:, 5);
%les donnees pour le test
X_test = data2(:, 1:4);
y_test = data2(:, 5);
%les donnes pour la validation
X_valid = data3(:, 1:4);
y_valid = data3(:, 5);

%% =========== Part 2: Initialisation des parametres =============
fprintf('\n  Initialisation des paramètres du réseau neuronal ...  \n')
%Ajouter seulement theta1 et theta2 car on a que 3 layer
Theta1 = randWeight_Team6(input_layer_size, hidden_layer_size);
Theta2 = randWeight_Team6(hidden_layer_size, num_labels);
nn_params = [Theta1(:) ; Theta2(:)];

%% =========== Part 3: entraînement de l'algorithme de réseau de neurone =============
fprintf('\n  entraînement de l algorithme de réseau neuronal  ...  \n')
options = optimset('MaxIter', 50);
lambda = 1;

costFunction = @(p) Cost_grad_Team6(nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);

[nn_params, cost] = fmincg_Team6(costFunction, nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
%% =========== Part 4: précision de l'algorithme NN =============

pred = predict_Team6(Theta1, Theta2, X_train);
prec = mean(double(pred == y_train) * 100)
fprintf('\n    Précision de l ensemble d entraînement == %f   \n', prec);
%-------------------------------------------------------------------------------%

pred = predict_Team6(Theta1, Theta2, X_test);
prec = mean(double(pred == y_test) * 100)
fprintf('\n    Précision de l ensemble de test == %f   \n', prec);
%----------------------------------------------------------------------------------%

pred = predict_Team6(Theta1, Theta2, X_valid);
prec = mean(double(pred == y_valid) * 100)
fprintf('\n    Précision de l ensemble de validation  == %f   \n', prec);

% (C) Tchikou Lakhdar G03---------Fin----------------------------
