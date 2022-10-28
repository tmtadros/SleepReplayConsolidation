%% MNIST catastrophic forgetting example, GIDO Class task
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../sleep'));
addpath(genpath('../utils'));


%%
num_epochs = 1:10;
memory = [100 1000 2000 20000];
num_trials = 5;
icarlaccuracy_normal = zeros(4,length(memory), length(num_epochs), num_trials);
icarlaccuracy_sleep = zeros(4,length(memory), length(num_epochs), num_trials);

for j = 1:num_trials
    task_order = randperm(5);
    for k = 1:length(num_epochs)
        for i = 1:length(memory)
            icarlaccuracy_normal(1,i,k,j) = run_icarl_cifar(task_order, 0, memory(i), num_epochs(k), 0, 0);
            icarlaccuracy_sleep(1,i,k,j) = run_icarl_cifar(task_order, 1, memory(i), num_epochs(k), 0, 0);
            icarlaccuracy_normal(2,i,k,j) = run_icarl_cifar(task_order, 0, memory(i), num_epochs(k), 0, 1);
            icarlaccuracy_sleep(2,i,k,j) = run_icarl_cifar(task_order, 1, memory(i), num_epochs(k), 0, 1);
            icarlaccuracy_normal(3,i,k,j) = run_icarl_cifar(task_order, 0, memory(i), num_epochs(k), 1, 0);
            icarlaccuracy_sleep(3,i,k,j) = run_icarl_cifar(task_order, 1, memory(i), num_epochs(k), 1, 0);
            icarlaccuracy_normal(4,i,k,j) = run_icarl_cifar(task_order, 0, memory(i), num_epochs(k), 1, 1);
            icarlaccuracy_sleep(4,i,k,j) = run_icarl_cifar(task_order, 1, memory(i), num_epochs(k), 1, 1);
        end
    end
    save('icarlcifar_accuracy_sleep', 'icarlaccuracy_normal', 'icarlaccuracy_sleep')
end
save('icarlcifar_accuracy_sleep', 'icarlaccuracy_normal', 'icarlaccuracy_sleep')
