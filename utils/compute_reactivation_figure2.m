function [f] = compute_reactivation(NNs, x,y, time_window)
% Correlation btw activity during awake and firing rates during sleep
[~, labels] = max(y, [], 2);

for i = 1:5
    nn1 = NNs{i}{1};   
    Snn = NNs{i}{2};

    nn_before = nnff(nn1, x, y);
    f = figure(i);
    
    for j = 2:nn_before.n
        % compute normalization factor
        activations = nn_before.a{j};
        R = corrcoef(activations);
        normalization_factor = mean(mean(R));

        subplot(1,3,j-1)
        correlations = zeros(200,10);
        pvals = zeros(200,10);
        for k = 1:10
            for l = 1:200
                indices = find(labels == k);
                activations = nanmean(nn_before.a{j}(indices,:));
                time = size(Snn.layers{j}.total_spikes, 1);
                
                time_ind = randi(time-time_window, 1);
                
                
                firingrates = nanmean(Snn.layers{j}.total_spikes(time_ind:time_ind+time_window,:));
                [corr_matrix, pval] = corrcoef(activations, firingrates);
                correlations(l,k) = corr_matrix(1,2)/normalization_factor;
                pvals(l,k) = pval(1,2);
            end
        end
        bar(0:9, nanmean(correlations)); hold on;

        er = errorbar(0:9, nanmean(correlations), nanstd(correlations));    
        er.Color = [0 0 0];                            
        er.LineStyle = 'none';  
        ylabel('Normalized correlation')
        xlabel('Digit')
        title(strcat('Reactivation in layer ', num2str(j)))
        set(gca, 'FontSize', 6)
        nanmean(correlations)
        nanmean(pvals)
        hold off;
    end
    
end
end

