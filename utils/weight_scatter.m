function [ f ] = weight_scatter( nn1, nn2, Snn, test_x, test_y, numneurons)
%Plots scatter plot of activitations vs. firing rate in spiking network
nn_firsttask = nnff(nn1, test_x, test_y);
nn_secondtask = nnff(nn2, test_x, test_y);

f = figure();
for i = 1:length(nn1.W)
    num_neurons = numneurons(i);
    [x,y] = size(nn1.W{i});

    % compute activations
    activations = mean(nn_firsttask.a{i+1});    
    activations2 = mean(nn_secondtask.a{i+1});
    firing_rates = Snn.layers{i+1}.sum_spikes;

    % sort activations
    [~, inds] = sort(firing_rates);
    neuron_inds = inds(end:-1:1);
    neuron_inds = neuron_inds(1:num_neurons);
    scatter(activations(neuron_inds), firing_rates(neuron_inds))
    rep_act1 = reshape(repmat(activations(neuron_inds),y,1)', num_neurons*y, 1);
    rep_act2 = reshape(repmat(activations2(neuron_inds), y,1)', num_neurons*y, 1);
    frs = reshape(repmat(firing_rates(neuron_inds), y, 1)', num_neurons*y,1);
    
    subplot(1,3,i)
    weights1 = reshape(nn1.W{i}(neuron_inds,:),num_neurons*y,1);
    weights2 = reshape(nn2.W{i}(neuron_inds,:), num_neurons*y, 1);
   
    %weights3 = reshape(Snn.W{i} - nn1.W{i}, x*y,1);
    %scatter(weights1, weights2, 40, frs)
    %scatter(rep_act1, frs)
    xlabel('Weights, train task 1')
    ylabel('Weights, train task 2')
    %zlabel('Spiking network firing rates')
    colormap(winter)
    colorbar()
    %caxis([-0.005 0.0005])
    title(strcat('Layer ', num2str(i)))
    set(gca, 'FontSize', 10)
end


end