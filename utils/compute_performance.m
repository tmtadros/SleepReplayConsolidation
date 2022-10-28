function overall_performance=compute_performance(NNs, X, Y)
%Plots the performance for each task seperately before and after sleep
%(tasks seen so far)
Snn = NNs{length(NNs)}{2};
[er, ~] = nntest(Snn, X, Y);
overall_performance = er*100;

%  performance