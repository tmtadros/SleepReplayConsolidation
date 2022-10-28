%% CUB200 catastrophic forgetting example, GIDO Class task
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));
percent_replay = [0.0 0.05 0.1 0.15 0.2 0.25];

%% 
num_trials = 5;
accuracy_sleept1 = zeros(length(percent_replay), num_trials);
accuracy_sleept2 = zeros(length(percent_replay), num_trials);
accuracy_nosleept1 = zeros(length(percent_replay), num_trials);
accuracy_nosleept2 = zeros(length(percent_replay), num_trials);
for i = 1:length(percent_replay)
  
    for j = 1:num_trials
        [accuracy_sleept1(i,j),accuracy_sleept2(i,j)] = run_sleep_with_replay(percent_replay(i), 1)
        [accuracy_nosleept1(i,j), accuracy_nosleept2(i,j)] = run_sleep_with_replay(percent_replay(i), 0)
    end
end

save('cub200_accuracy', 'accuracy_sleept1', 'accuracy_nosleept1', 'accuracy_sleept2', 'accuracy_nosleept2')
