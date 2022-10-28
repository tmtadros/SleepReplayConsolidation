function [ f ] = compute_neural_reactivation( ann, snn, X, y, numNeurons )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

nn_temp = nnff(ann, X, y); % for activations in ANN
[~, labels] = max(y, [], 2);


frs = zeros(length(nn_temp.a)-1, 1);
fr_stds = zeros(length(nn_temp.a)-1, 1);

for i = 1:length(nn_temp.a)-1 % number of layers
    for j = 1:10 % number of classes in dataset
        activations = nn_temp.a{i}(labels == j,:);
        spikes = snn.layers{i}.sum_spikes;
        firing_rates = zeros(length(activations), 1);
        for k = 1:length(activations)
            [~, high_firing_inds] = sort(activations(k,:), 'descend');
            high_firing_inds = high_firing_inds(1:numNeurons);
            firing_rates(k) = mean(spikes(high_firing_inds));
        end
        frs(i,j) = mean(firing_rates);
        fr_stds(i,j) = std(firing_rates);
 
    end
end


f = figure();
for i = 1:length(nn_temp.a)-1
    subplot(2,2,i)
    
    bar(frs(i,:)); hold on;
    er = errorbar(1:10,frs(i,:),fr_stds(i,:));    
    er.Color = [0 0 0];                            
    er.LineStyle = 'none'; 
    set(gca,'XTickLabel',{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'});
    xlabel('Digit class')
    ylabel('Average firing rate')
    title(strcat('Layer ', num2str(i)))
    hold off
end
suptitle('Average firing rate of class-specific neurons during sleep')


end

