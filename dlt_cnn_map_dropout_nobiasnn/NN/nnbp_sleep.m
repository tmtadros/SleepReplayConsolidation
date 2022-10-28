function nn = nnbp_sleep(nn)

%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
   

    %feedforward pass
    for l = 2 : n-1
        post = nn.a{l};
        pre = nn.a{l-1};
        nn.W{l-1}(post==1,pre==1) = nn.W{l-1}(post==1,pre==1) + ...
            nn.inc * sigmoid(nn.W{l-1}(post==1,pre==1));
        nn.W{l-1}(post==1,pre==0) = nn.W{l-1}(post==1,pre==0) - ...
            nn.dec * sigmoid(nn.W{l-1}(post==1,pre==0));

        % Reset
        nn.v{l}(post) = 0;

    end
        
end

function y = sigmoid(x)
    y = 2* (1.0 - (1.0 ./ (1 + exp(-x/0.001))));
end