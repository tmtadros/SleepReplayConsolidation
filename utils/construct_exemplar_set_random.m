function [S] = construct_exemplar_set_random(X, m,nn)
   inds = randperm(length(X));
   S = X(inds(1:m),:);
end