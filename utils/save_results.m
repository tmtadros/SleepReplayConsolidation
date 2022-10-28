function save_results( fsc, Ann, Snn, testX, testY )

%% Save conf matrix
[~, labels] = max(testY, [], 2);

awakeConfMat = zeros(10,10);
sleepConfMat = zeros(10,10);
for i = 1:10
    indices = find(labels == i);
    outputLabels = nnpredict(Ann, testX(indices,:));
    sleepoutputLabels = nnpredict(Snn, testX(indices,:));
    for j = 1:10
        awakeConfMat(i,j) = sum(outputLabels == j);
        sleepConfMat(i,j) = sum(sleepoutputLabels == j);
    end
end

fname = [fsc, 'awakeConfMat'];
save(fname, 'awakeConfMat') 

fname = [fsc, 'sleepConfMat'];
save(fname, 'sleepConfMat') 

%% save weights
for i = 1:3
    awake_weights = Ann.W{i};
    sleep_weights = Snn.W{i};
    save([fsc, '-awakeW', num2str(i)],'awake_weights');
    save([fsc, '-sleepW', num2str(i)],'sleep_weights');
end

%% save firing rates during sleep
for l = 1 : numel(Snn.size)
    spikes_sum = Snn.layers{l}.sum_spikes;
    save([fsc, '-fr', num2str(l)],'spikes_sum');
end


