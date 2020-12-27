function [X_normalise, m, sigma] = normaliser_Team02(X)
X_normalise = X;
m = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
m = mean(X_normalise);
sigma = std(X_normalise);
tfmu = X_normalise - repmat(m,length(X_normalise),1);
tfstd = repmat(sigma,length(X_normalise),1);
X_normalise = tfmu ./ tfstd;
end