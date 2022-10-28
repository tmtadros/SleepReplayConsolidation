%% MNIST catastrophic forgetting example, GIDO Class task
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));

%% load mnist_uint8 and create task;
load mnist_uint8;
train_x1 = double(train_x) / 255;
test_x1  = double(test_x)  / 255;
train_y1 = double(train_y);
test_y1  = double(test_y);

load fashionmnist_uint8;
train_x2 = double(train_x) / 255;
test_x2  = double(test_x)  / 255;
train_y2 = double(train_y);
test_y2  = double(test_y);

accuracy = zeros(3,5);
for j = 1:5
    %% Set up neural network
    %Initialize net
    nn = nnsetup([784 1200 1200 10]);
    % Rescale weights for ReLU
    for i = 2 : nn.n   
        % Weights - choose between [-0.1 0.1]
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
        nn.vW{i - 1} = zeros(size(nn.W{i-1}));
    end

    % ReLU Train
    % Set up learning constants
    nn.activation_function = 'relu';
    nn.output ='relu';
    nn.learningRate = 0.1;
    nn.momentum = 0.5;
    nn.dropoutFraction = 0.25;
    nn.learn_bias = 0;
    opts.numepochs =  2;
    opts.batchsize = 100;

    ideal_nn = nntrain(nn, [train_x1; train_x2], [train_y1; train_y2], opts);
    
    control_nn1 = nntrain(nn, train_x1, train_y1, opts);
    control_nn2 = nntrain(nn, train_x2, train_y2, opts);
    
    
    x = [227.437205 0.050052 0.005592 4.700579 7.750487 7.055086 9.284002 6665.956743 12468.269487
    ];
    inp_rate=x(1);
    inc=x(2); 
    dec=x(3);
    b1=x(4); % 1.1
    b2 =x(5);
    b3 = x(6);
    alpha_scale = x(7);
    l1 = x(8);
    l2 = x(9);

    t_opts = struct;
    t_opts.t_ref        = 0.000;
    t_opts.threshold    =   1.0;
    t_opts.dt           = 0.001;
    t_opts.duration     = 0.035;
    t_opts.report_every = 0.001;
    t_opts.max_rate     = inp_rate;
    %t_opts.max_rate=16;

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
    nn1 = nntrain(nn_pointer, train_x1, train_y1, opts);
    nn_pointer = nn1;
    labels{1} = 'Train Task: MNIST';
    labels{2} = 'Sleep';

    % sleep
    sleep_period = uint8(l1);
    sleep_input = create_masked_input(train_x1, sleep_period, 10);
    [~, norm_constants] = normalize_nn_data(nn_pointer, train_x1);
    sleep_opts.alpha = norm_constants*alpha_scale; %*2.25;

    % Run NREM
    Snn = sleepnn_old(nn_pointer, sleep_period, t_opts, sleep_opts, ...
                  sleep_input'); % , threshold_scales

    NNs{1} = {nn1, Snn};
    nn1 = nntrain(Snn, train_x2, train_y2, opts);
    nn_pointer = nn1;
    labels{3} = 'Train Task: fashionMNIST';
    labels{4} = 'Sleep';

    % sleep
    sleep_period = uint8(l2);
    training_data = cat(1, train_x1, train_x2);
    training_data = train_x1;
    sleep_input = create_masked_input(training_data, sleep_period, 10);
    [~, norm_constants] = normalize_nn_data(nn_pointer, training_data);
    sleep_opts.alpha = norm_constants*alpha_scale; %*2.25;

    % Run NREM
    Snn = sleepnn_old(nn_pointer, sleep_period, t_opts, sleep_opts, ...
                  sleep_input'); % , threshold_scales

    % test accuracy on 3 networks
    [er, bad] = nntest(ideal_nn, [test_x1; test_x2], [test_y1; test_y2]);
    accuracy(1,j) = (1-er);
    [er, bad] = nntest(control_nn2, [test_x1; test_x2], [test_y1; test_y2]);
    accuracy(2,j) = (1-er);
    [er, bad] = nntest(Snn, [test_x1; test_x2], [test_y1; test_y2]);
    accuracy(3,j) = (1-er);
end
