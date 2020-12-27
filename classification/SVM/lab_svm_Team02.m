clear; close all; clc;
 % preparation du dataset
data = load('PiiishingData.txt');
X = data(:,1:9);
y = data(:,10);

% Diviser nos données sur 2 (70% apprentissage et 30% test)
% données d'apprentissage
X_train = X(1:506,:);
y_train = y(1:506,:);

% données de test
X_test = X(507:724,:);
y_test = y(507:724,:);

% entrainnement
model = fitcecoc(X_train, y_train);
% prediction 
y_prediction = predict(model, X_test);
% Afficher la précision (la similarité entre y_restant et y qu'on a prédicté)
fprintf('Précision : %f \n', mean(double(y_test == y_prediction))*100);


