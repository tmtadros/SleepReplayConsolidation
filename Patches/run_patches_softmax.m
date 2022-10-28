%% Train a neural network on 4 patches
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../sleep'));
addpath(genpath('../utils'));

%% Set up patches
num_images = 4;
width = 11;
overlaps = [12];
num_trials=1;
acc_after_task1 = zeros(length(overlaps),num_trials);
acc_after_task2 = zeros(length(overlaps),num_trials);
acc_after_sleep1 = zeros(length(overlaps),num_trials);
%acc_after_sleep2 = zeros(length(overlaps),5);

t1sum = zeros(num_trials,3);
t2sum = zeros(num_trials,3);
for k = 1:num_trials
for j = 1:length(overlaps)
    [X,y] = create_permutations(width, overlaps(j), num_images);

    nn = nnsetup([121 4]);
    for i = 2 : nn.n   
        % Weights - choose between [-0.1 0.1]
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
        nn.vW{i - 1} = zeros(size(nn.W{i-1}));
    end

        % Set up learning constants
        nn.activation_function = 'relu';
        nn.output ='softmax';
        nn.learningRate = 0.1;
        nn.momentum = 0.5;
        %nn.dropoutFraction = 0.5;
        nn.learn_bias = 0;
        opts.numepochs =  4;
        opts.batchsize = 2;

        t_opts = struct;
        t_opts.t_ref        = 0.000;
        t_opts.threshold    =   1.0;
        t_opts.dt           = 0.001;
        t_opts.duration     = 0.035;
        t_opts.report_every = 0.001;
        t_opts.max_rate     =  64;
        %t_opts.max_rate=16;

        sleep_opts.beta = [0.95 0.65 0.35]*1.1;
        % sleep_opts.alpha = [2.50 4.0 7.5]*1.25; % -- This is reset later
        sleep_opts.decay = 0.999; 

        sleep_opts.W_inh=0.0;
        sleep_opts.normW = 0;
        %sleep_opts.beta = [0.6 0.3 0.5];
        %sleep_opts.decay = 0.999;
        %sleep_opts.alpha = [2.50 2.5 2.5];

        sleep_opts.inc = 0.0035;
        sleep_opts.dec = 0.0002;

        sleep_opts.delta_min=1000;
        sleep_opts.delta_max=1000;

        sleep_opts.theta = 0.0;
        numiterations=30000;

        % Train - takes about 15 seconds per epoch on my machine
        nn1 = nntrain(nn, X(1:2,:), y(1:2,:), opts);
        % Test - should be 98.62% after 15 epochs
        [er, train_bad] = nntest(nn1, X, y);
        acc_after_task1(j,k) = (1-er)*100;

        %sleep_period = numiterations;
        %sleep_input = mean(X(1:2,:), 1);
        %sleep_input = repmat(sleep_input, sleep_period,1);
        %[norm_nn, norm_constants] = normalize_nn_data(nn1, X(1:2,:));
        %sleep_opts.alpha = norm_constants*4.25; %*2.25;
        %Snn1 = sleepnn(nn1, sleep_period, t_opts, sleep_opts, sleep_input'); % , threshold_scales
        Snn1 = nn1;

        % Test - should be 98.62% after 15 epochs
        %[er, train_bad] = nntest(Snn1, X, y);
        %acc_after_sleep1(j,k) = (1-er)*100;    
        opts.numepochs =  4;

        nn2 = nntrain(Snn1, X(3:4,:), y(3:4,:), opts);
        % Test - should be 98.62% after 15 epochs
        [er, train_bad] = nntest(nn2, X, y);
        acc_after_task2(j,k) = (1-er)*100;
        task1_acc = (1-er)*100;
        
        sleep_period = numiterations;
        sleep_input = mean(X, 1);
        sleep_input = repmat(sleep_input, sleep_period,1);
        [norm_nn, norm_constants] = normalize_nn_data(nn2, X);
        sleep_opts.alpha = norm_constants*4.25; %*2.25;
        
        % Run NREM
        Snn2 = sleepnn_old(nn2, sleep_period, t_opts, sleep_opts, sleep_input'); % , threshold_scales
       
        % Test - should be 98.62% after 15 epochs
        [er, train_bad] = nntest(Snn2, X, y);
        acc_after_sleep2(j,k) = (1-er)*100;
        
        [t1s, t2s] = cf_analysis({nn1.W{1}, nn2.W{1}, Snn2.W{1}}, X);
        t1sum(k,:) = t1s;
        t2sum(k,:) = t2s;

end
end

%%

meanacc_nosleep = mean(t1sum', 2);
meanacc_sleep = mean(t2sum', 2);
err_nosleep = std(t1sum', [], 2);
err_sleep = std(t2sum', [], 2);

y = [meanacc_nosleep, meanacc_sleep];
err = [err_nosleep, err_sleep];

figure;
hb = bar(y); hold on;
hb(1).FaceColor = 'k';
hb(2).FaceColor = 'r';

ngroups = 3;
nbars = size(y, 2);
% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    er = errorbar(x, y(:,i), err(:,i), '.');
    er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
end

legend('T1 Input', 'T2 Input')
ylabel('Classification Accuracy')
xlabel('Num. Training Epochs')
xticklabels({'Train T1', 'Train T2', 'Sleep'})
set(gca,'FontSize',20)

plot_weights_hist_new({nn1.W{1}, nn2.W{1}, Snn2.W{1}}, X)
