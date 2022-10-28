%% CUB catastrophic forgetting example, GIDO Class task
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));
%rng('default');
%rng(1000);
%% Load in CIFAR data and normal
load 'cifar_data_1000_256_tinyimagenet.mat'
train_x = train_x / max(train_x(:));
test_x = test_x / max(test_x(:));
train_x = double(train_x);
test_x = double(test_x);
[tasks, train_y] = create_class_task(train_x, train_y);
[test_tasks, test_y] = create_class_task(test_x, test_y);

%% Build neural network architecture
accuracy = zeros(3,5);
for j = 1:5
    %Initialize net
    nn = nnsetup([2048 1028 256 10]);
    % Rescale weights for ReLU
    for i = 2 : nn.n   
        % Weights - choose between [-0.1 0.1]
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.05 * 2;
        nn.vW{i - 1} = zeros(size(nn.W{i-1}));
    end
    task_order = randperm(5);

    % ReLU Train
    % Set up learning constants
    nn.activation_function = 'relu';
    nn.output ='relu';
    nn.learningRate = 0.1;
    nn.momentum = 0.5;
    nn.dropoutFraction = 0.2;
    nn.learn_bias = 0;
    opts.numepochs = 4;
    opts.batchsize = 100;

    ideal_nn = nntrain(nn, train_x, train_y, opts);
    control_nn = nn;
    for ii = 1:5
        % Train on taski
        i=task_order(ii);
        indices = find(tasks == i);
        control_nn = nntrain(control_nn, train_x(indices,:), train_y(indices,:), opts);
    end

    x = [392.265660 0.021282 0.002064 9.738645 1.177364 12.895054 2.635832
    ];
    x = [453.020604 0.078129 0.003005 8.070491 0.907842 1.728840 4.131109];

    inp_rate=x(1);
    inc=x(2); 
    dec=x(3);
    b1 = x(4);
    b2 = x(5);
    b3 = x(6);
    alpha_scale = x(7);

    t_opts = struct;
    t_opts.t_ref        = 0.000;
    t_opts.threshold    =   1.0;
    t_opts.dt           = 0.001;
    t_opts.duration     = 0.035;
    t_opts.report_every = 0.001;
    t_opts.max_rate     = inp_rate;

    sleep_opts.beta = [b1 b2 b3];
    sleep_opts.decay = 0.999; 
    sleep_opts.W_inh = 0.0;
    sleep_opts.normW = 0;
    %sleep_opts.beta = {0.6, 0.3, 0.5};
    %sleep_opts.decay = 0.999;
    %sleep_opts.alpha = [2.50 2.5 2.5];
    sleep_opts.inc = inc;
    sleep_opts.dec = dec;
    sleep_opts.DC = 0.0;

    NNs = {};
    labels = cell(10,1);
    nn_pointer = nn;
    %% Train all tasks sequentially
    %accuracy = zeros(3,6);
    plot_i = 1;
    numiterations = 15000;
    layers = [];

    %% TODO: get task order from permutation
    %task_order = [1,4];
    for ii = 1:5
        i=task_order(ii);

        % Train on taski
        indices = find(tasks == i);
        if ii <= 5
            nn1 = nntrain(nn_pointer, train_x(indices,:), train_y(indices,:), opts);
        else
            inds2 = find(ismember(tasks, task_order(1:ii-1)));
            tr_x = vertcat(train_x(indices,:), train_x(inds2(1:2000),:));
            tr_y = vertcat(train_y(indices,:), train_y(inds2(1:2000),:));
            nn1 = nntrain(nn_pointer, tr_x, tr_y, opts);
        end
        nn_pointer = nn1;

        labels{plot_i} = strcat('Train Task: ', num2str(i));

        % sleep
        indices = find(ismember(tasks, task_order(1)));
        sleep_period = numiterations;

        sleep_input = create_masked_input_CIFAR(train_x(indices,:), sleep_period, 2047);

        [~, norm_constants] = normalize_nn_data(nn_pointer, train_x(indices,:));
        sleep_opts.alpha = norm_constants * alpha_scale;

        % Run NREM
        Snn = sleepnn_old(nn_pointer, sleep_period, t_opts, sleep_opts, ...
                      sleep_input'); % , threshold_scales

        nn_pointer = Snn;
        for k = 1:length(layers)
           Snn.W{layers(k)} = nn1.W{layers(k)};  
        end
        NNs{ii} = {nn1, Snn};
        %labels{plot_i + 1} = "Sleep";
        %confFig = confmat(nn1, Snn, test_x, test_y);
        %[f1, f2, f3, f4] = plot_firing_rates(Snn);
        %[w1, w2] = plot_weight_distributions(nn1, Snn);  
    end
    [er, bad] = nntest(ideal_nn, test_x, test_y);
    accuracy(1,j) = 1-er;
    [er, bad] = nntest(control_nn, test_x, test_y);
    accuracy(2,j) = 1-er;
    [er, bad] = nntest(Snn, test_x, test_y);
    accuracy(3,j) = 1-er;
end
