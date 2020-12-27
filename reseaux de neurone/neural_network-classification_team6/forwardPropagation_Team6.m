function a3 = forwardPropagation_Team6( x,m,Theta1,Theta2 )
%% =========== 1-Implementation de forward propagation=================================================================
%ajouter le (bias) a la premiere colone de x 
                
                        % ======= LAYER 2 =======
z2 = x*Theta1';             %sigmoid input: z = x*thetha1
a2 = sigmoid_Team6(z2);   %activation unit: a2=g(z)= sigmoid(z) 
a2 = [ones(m, 1), a2];      %ajouter le (bias) a la premiere colone de a2 
                         % ======= LAYER 3 =======
z3 = a2*Theta2';
a3 = sigmoid_Team6(z3);
end

