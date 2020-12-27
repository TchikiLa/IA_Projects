clear; close all; clc;
 % preparation du dataset
data = load('PiiishingData.txt');
X = data(:,1:9);
y = data(:,10);

% Diviser nos donn�es sur 2 (70% apprentissage et 30% test)
% donn�es d'apprentissage
X_train = X(1:506,:);
y_train = y(1:506,:);

% donn�es de test
X_test = X(507:724,:);
y_test = y(507:724,:);

% entrainnement
model = fitcecoc(X_train, y_train);
% prediction 
y_prediction = predict(model, X_test);
% Afficher la pr�cision (la similarit� entre y_restant et y qu'on a pr�dict�)
fprintf('Pr�cision : %f \n', mean(double(y_test == y_prediction))*100);


