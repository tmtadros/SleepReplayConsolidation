function [ f ] = activity_scatter( nn1, nn2, Snn, test_x, test_y, test_tasks, task1, task2)
%Plots scatter plot of activitations vs. firing rate in spiking network

nn_firsttask = nnff(nn1, test_x(test_tasks==task1,:), test_y(test_tasks==task1,:));
nn_secondtask = nnff(nn2, test_x(test_tasks==task2,:), test_y(test_tasks==task2,:));
f = figure();
for i = 2:Snn.n
    subplot(1,3,i-1)
    activations = mean(nn_firsttask.a{i});
    activations2 = mean(nn_secondtask.a{i});
    
    firing_rates = Snn.layers{i}.sum_spikes;
    
    scatter(activations, activations2, 50, firing_rates)
    xlabel('Neural network activation, First task')
    ylabel('Neural network activation, Second task')
    colormap(winter)
    colorbar()
    caxis([0 10])
    title(strcat('Layer ', num2str(i)))
end


end

