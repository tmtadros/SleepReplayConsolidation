function [ beta ] = create_thresholds( nn1, test_x, test_y, binsize, scale)
%Creates the thresholds for each neuron to be used in spiking network
% during sleep
% Params:
% nn1, neural network file trained using backprop
% test_x, testing examples of previously trained task used to compute
% activations

nn_act = nnff(nn1, test_x, test_y);

thresholds = 0:1:binsize;
thresholds = thresholds/binsize;

beta = {};
for l = 2:numel(nn1.size)
    %beta{l} = zeros(nn1.size(l));
    activations = mean(nn_act.a{l});
    num_neurons = length(activations);
    neuronal_thresholds = zeros(num_neurons,1);
    num_examples_per_bin = num_neurons/binsize;
    [~, inds] = sort(activations);
    for j = 1:binsize
       neuronal_thresholds(inds((j-1)*num_examples_per_bin+1:j*num_examples_per_bin)) = ...
           repmat(thresholds(j), num_examples_per_bin, 1)' * scale(l-1);
    end
    beta{l-1} = neuronal_thresholds;
end
end

