function [ f ] = digit_reactivation( ann, snn, X, y )
%Computes the amount of overlap by digit class between activation during
%training in ANN and spiking activity in SNN


nn_temp = nnff(ann, X, y); % for activations in ANN
[~, labels] = max(y, [], 2);

correlations = zeros(10,length(nn_temp.a));
correlationsSTD = zeros(10,length(nn_temp.a));

for i = 1:length(nn_temp.a) % number of layers
    for j = 1:10 % number of classes in dataset
        activations = nn_temp.a{i}(labels == j,:);
        spikes = snn.layers{i}.sum_spikes;
        
        corrs = zeros(length(activations),1);
        for k = 1:length(activations)
            corr2d = corrcoef(activations(k,:), spikes);
            corrs(k) = corr2d(2,1);
        end
        
            
        correlations(j,i) = mean(corrs);
        correlationsSTD(j,i) = std(corrs);
    end
end

f = figure();
for i = 1:length(nn_temp.a)
    subplot(2,2,i)
    
    bar(correlations(:,i)); hold on;
    er = errorbar(1:10,correlations(:,i),correlationsSTD(:,i));    
    er.Color = [0 0 0];                            
    er.LineStyle = 'none'; 
    set(gca,'XTickLabel',{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'});
    xlabel('Digit class')
    ylabel('Correlation')
    title(strcat('Layer ', num2str(i)))
    hold off
end
suptitle('Correlation between ANN activation and spiking activity')

end



