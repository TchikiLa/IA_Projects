function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
W = zeros(L_out, 1 + L_in);
EPSILON=sqrt(6)./(sqrt(L_in)+sqrt(L_out+1));
W = rand(L_out, 1 + L_in) * 2 * EPSILON-EPSILON;

end
