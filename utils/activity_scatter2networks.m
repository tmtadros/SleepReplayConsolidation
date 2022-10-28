function [ f ] = activity_scatter2networks( nn1, nn2, test_x, test_y)
%Plots scatter plot of activitations vs. firing rate in spiking network

nn_firsttask = nnff(nn1, test_x, test_y);
nn_secondtask = nnff(nn2, test_x, test_y);
f = figure();
for i = 2:nn1.n
    subplot(1,3,i-1)
    activations = mean(nn_firsttask.a{i});
    activations2 = mean(nn_secondtask.a{i});
    
    scatter(activations, activations2)
    xlabel('Neural network activation, First network')
    ylabel('Neural network activation, Second network')
    %zlabel('Spiking network firing rates')

    title(strcat('Layer ', num2str(i)))
end


end