function [S] = construct_exemplar_set2(X, m, nn)
    nn = nnff(nn, X, zeros(size(X,1), nn.size(end)));
    
    activations = nn.a{length(nn.a)-1};
    mu = mean(activations,1);
    S = zeros(m, size(X,2));
    indices = [];
    
    [out, index] = sort(activations * mu');
    indices(1) = index(1);
    S(1,:) = X(index(1),:);
    for i = 2:m
        exemplar_sum = sum(activations(indices,:),1);
        phi = activations;
        
        mu_p = 1.0/(i) * (phi + exemplar_sum);
        [~,index] = sort((sum(((mu - mu_p).^2),2)).^(1/2));
        
        j = 1;
        while ismember(index(j), indices)
            j = j + 1;
        end
        S(i,:) = X(index(j),:);
        indices(i) = index(j);
    end
    %[out, index] = sort(activations * mean_activation');
    
    %S = X(index(length(index)-m+1:end),:);

end