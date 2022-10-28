%% MNIST catastrophic forgetting example, GIDO Class task
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));

num_epochs = [10 20 40];
num_trials = 5;
accuracy_sleep = zeros(length(num_epochs), num_trials);
accuracy_nosleep = zeros(length(num_epochs), num_trials);

for i = 1:num_trials
   accuracy_sleep(1,i) = run_reduced_training10(1);
   accuracy_nosleep(1,i) = run_reduced_training10(0);
   accuracy_sleep(2,i) = run_reduced_training20(1);
   accuracy_nosleep(2,i) = run_reduced_training20(0);
   accuracy_sleep(3,i) = run_reduced_training40(1);
   accuracy_nosleep(3,i) = run_reduced_training40(0);
end    

save('reduced_mnist_accuracy', 'accuracy_sleep', 'accuracy_nosleep')
