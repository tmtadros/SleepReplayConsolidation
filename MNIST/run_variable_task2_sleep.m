%% MNIST catastrophic forgetting example, GIDO Class task
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));


%%
task2_training = 5:5:40;

num_trials = 1;
accuracy_task1= zeros(length(task2_training), num_trials);
accuracy_task2= zeros(length(task2_training), num_trials);

for i = 1:length(task2_training)
    for j = 1:num_trials
        task_order = [1 2 3 4 5];
        [t1acc, t2acc] = run_sleep_with_variable_task2_length(task_order, task2_training(i));
        accuracy_task1(i,j) = t1acc;
        accuracy_task2(i,j) = t2acc;
    end
end


save('mnist_accuracy_variabletask2_length', 'accuracy_task1', 'accuracy_task2')
