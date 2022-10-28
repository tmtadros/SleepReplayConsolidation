function [er, bad] = nntest_iCarl(nn, x, y, S, Sy)
    labels = nnpredict_iCarl(nn, x, S, Sy);
    [dummy, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
end
