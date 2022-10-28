%% Cross modal catastrophic forgetting example, GIDO Class task
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));
percent_replay = [0.0 0.01 0.025 0.05 0.1 0.25 0.5];

%% 
num_trials = 5;
accuracy_sleep = zeros(length(percent_replay), num_trials);
accuracy_nosleep = zeros(length(percent_replay), num_trials);
for i = 1:length(percent_replay)
  
    for j = 1:num_trials
        task_order = [1 2 3 4 5];
        accuracy_sleep(i,j) = run_sleep_with_replay(task_order, percent_replay(i), 1);
        accuracy_nosleep(i,j) = run_sleep_with_replay(task_order, percent_replay(i), 0);
    end
end

save('crossmodal_accuracy_optimized', 'accuracy_sleep', 'accuracy_nosleep')