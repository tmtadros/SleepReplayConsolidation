function nn = nnff_sleep(nn, x)

%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);
    
    %x = [ones(m,1) x];
    nn.a{1} = x;
    for i = 2:n-1
        nn.a{i} = zeros(1,nn.size(i));
    end

    %feedforward pass
    for i = 2 : n-1
        nn.v{i} = nn.decay*nn.v{i} + (nn.a{i - 1}' * (nn.W{i - 1}'*nn.w_scale(i-1)))';
        nn.a{i} = nn.v{i} > nn.threshold(i-1);

    end
        
end
