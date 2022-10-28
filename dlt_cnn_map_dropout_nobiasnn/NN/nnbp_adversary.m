function [ derivative ] = nnbp_adversary(nn, x)
%NNBP_adversary performs backpropagation with respect to input x 
% based on fast gradient descent method in Goodfellow et al.

    n = nn.n;
    switch nn.output
        case 'sigm'
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
        case {'softmax','linear', 'relu'}
            d{n} = - nn.e;
    end
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
                d_act = nn.a{i} .* (1 - nn.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
            case 'relu'
                d_act = single(nn.a{i} > 0);
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn.W{i}) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1} * nn.W{i}) .* d_act;
        end
    end

    derivative = d{i} * nn.W{1};
end
