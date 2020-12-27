% (C) Tchikou Lakhdar G03---------Debut----------------------------
clear ; close all; clc

%% =========== Part 0: Configurer les parametre de notre architecture  =============
input_layer_size  = 12;  % 12 Input variables
hidden_layer_size = 14;   % 14 unitées cachées 
num_labels = 8;          % 8 unitées de sorites, de 1 au 8   

%% =========== Part 1: Chargement des données =============

% Load Training Data
fprintf('\n  Chargement des données ...\n');
data = load('ff.csv');
x = data(:, 1:12);
y = data(:, 13);
figure;
plot(y, 'rx', 'MarkerSize', 10);
xlabel('DATA');
m = size(x, 1);
y = log(y+1);
siz_y = length(y);
for i = 1:siz_y
    if y(i)==0  
        y(i)=8;
    end
    if y(i)>0  && y(i)<=1 
        y(i)=1;
    end
    if y(i)>1  && y(i)<=2 
        y(i)=2;
    end
    if y(i)>2  && y(i)<=3 
        y(i)=3;
    end
    if y(i)>3  && y(i)<=4 
        y(i)=4;
    end
    if y(i)>4  && y(i)<=5 
        y(i)=5;
    end
    if y(i)>5  && y(i)<=6 
        y(i)=6;
    end
    if y(i)>6  && y(i)<=7 
        y(i)=7;
    end
end
figure;
plot(y, 'rx', 'MarkerSize', 10);
xlabel('DATA');
%% =========== Part 2: Initialisation des parametres =============
fprintf('\n  Initialisation des paramètres de réseau neuronal ...  \n')
%Ajouter que theta1 et theta2 car on a que 2 layer
Theta1 = NNrandinittheta_Team9(input_layer_size, hidden_layer_size);
Theta2 = NNrandinittheta_Team9(hidden_layer_size, num_labels);

nn_params = [Theta1(:) ; Theta2(:)];
%% =========== Part 3: entraînement de l'algorithme de réseau de neurone =============
fprintf('\n  entraînement de l algorithme de réseau neuronal  ...  \n')
options = optimset('MaxIter', 50);
lambda = 1;
%les donne de traitement
X_train = x(1:361, :);
y_train = y(1:361, :);
%les donnes pour la validation
X_valid = x(361+1:465, :);
y_valid= y(361+1:465, :);
%les donnees pour le test
X_test = x(465+1:end, :);
y_test=y(465+1:end, :);

costFunction = @(p) NNcostgrad_Team9(nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);

[nn_params, cost] = NNfmincg_Team9(costFunction, nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%% =========== Part 4: précision de l'algorithme NN =============
pred = NNpredict_Team9(Theta1, Theta2, X_train);
%rndre les valeur egales à 8 à 0 pour maintenir un resultat juste
for i = 1:length(pred)
    if pred(i) == 8
        pred(i)=0;
    end
end
prec = mean(double(pred == y_train) * 100)
fprintf('\n    Précision de l ensemble d entraînement == %f   \n', prec*10);

%-------------------------------------------------------------------------------%
pred = NNpredict_Team9(Theta1, Theta2, X_valid);
%rndre les valeur egales à 8 à 0 pour maintenir un resultat juste
for i = 1:length(pred)
    if pred(i) == 8
        pred(i)=0;
    end
end
prec = mean(double(pred == y_valid) * 100)
fprintf('\n    Précision de l ensemble de validation  == %f   \n', prec*10);
%----------------------------------------------------------------------------------%

pred = NNpredict_Team9(Theta1, Theta2, X_test);
%rndre les valeur egales à 8 à 0 pour maintenir un resultat juste
for i = 1:length(pred)
    if pred(i) == 8
        pred(i)=0;
    end
end
prec = mean(double(pred == y_test) * 100)
fprintf('\n    Précision de l ensemble de test == %f   \n', prec*10);
% (C) Tchikou Lakhdar G03---------Fin----------------------------
