clear ; close all; clc;
data = load('PiiishingData.txt');
X = data(:, [1: 9]);
y = data(:, 10);
[X muu sigma] = normaliser_Team02(X);% Normalisation des donn�es 
% les labels des classes 
% 0 : Suspect
% -1 : Phishing
% 1 : Legitime
labels = size(unique(y),1);
% Division des donn�es en deux parties 70% Pour l'apprentissage et 30% pour le test
Xtrain = data(1:392, [1: 9]);
Ytrain = data(1:392, 10);
Xtest = data(393:561, [1: 9]);
Ytest = data(393:561, 10);
alpha = 0.1;% Initialisation de taux d'apprentissage
iter = 500;% Initialisation de nombre d'it�rations
% Initialisation de param�tre de r�gularisation
lambda = 0.001;% Calcul de theta, les predictions, et cost
[theta,predictions, cost] = prediction_Team02(Xtrain,Ytrain,Xtest,labels,alpha,iter,lambda);
% Plot de la fonction cost
plot(1:iter,cost,'-b');
fprintf('Pr�cision : %f \n', mean(double(Ytest == predictions))*100);% Calcul de la pr�cision
