% (C) Tchikou Lakhdar G03---------Debut----------------------------
function g = NNsigmoidGradient_Team9(z)
%la fonction sigmoidGradient calcule le "G-prime derivitive terms ou le
%gradient du sigmoid 
g = zeros(size(z));
g = NNsigmoid_Team9(z).*(1-NNsigmoid_Team9(z));% g'(Z(l)) = a(l) .* (1-a(l)) // 
                                % ou a(l) = g(z(l)) = sigmoid(z(l))
end
% (C) Tchikou Lakhdar G03---------Fin----------------------------