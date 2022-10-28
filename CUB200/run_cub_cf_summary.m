%% CUB catastrophic forgetting example
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
accuracy = zeros(3,2,5);

train_set = load('cub200_resnet50_train.mat');
test_set = load('cub200_resnet50_train.mat');

train_x = train_set.X;
train_y = squeeze(train_set.y);
test_x = test_set.X;
test_y = squeeze(test_set.y);

%% Set up CF tasks
indices = randperm(5994);
train_y = train_y(indices);
train_x = train_x(indices,:);
trainlabels = train_y;
testlabels = test_y;

% y is a vector of labels
train_y_oh = zeros( size( train_y, 1 ), 200 );
test_y_oh = zeros( size( test_y, 1 ), 200);
% assuming class labels start from one
for i = 1:200
    train_rows = train_y == i;
    test_rows = test_y == i;
    train_y_oh( train_rows, i ) = 1;
    test_y_oh( test_rows,i) = 1;
end

task1_train_x = train_x(trainlabels<=100,:);
task2_train_x = train_x(trainlabels>100,:);
task1_train_y = train_y_oh(trainlabels<=100,:);
task2_train_y = train_y_oh(trainlabels>100,:);

task1_test_x = test_x(testlabels<=100,:);
task2_test_x = test_x(testlabels>100,:);
task1_test_y = test_y_oh(testlabels<=100,:);
task2_test_y = test_y_oh(testlabels>100,:);

%% Train network
for j = 1:5
    %Initialize net
    nn = nnsetup([2048 350 300 200]);
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
    opts.numepochs =  50;
    opts.batchsize = 100;
    
    inp_rate=32;
    inc=0.01; 
    dec=0.001;
    b1 = 1;
    b2 = 1;
    b3 = 1;
    alpha_scale = 1.0;
    sleep_dur = 5000;
    task2_LR = 0.01;
    
    ideal_nn = nntrain(nn, train_x(1:5800,:), train_y_oh(1:5800,:), opts);
    control_nn1 = nntrain(nn, task1_train_x(1:2900,:), task1_train_y(1:2900,:), opts);
    
    % Train on task1
    nn1 = nntrain(nn, task1_train_x(1:2900,:), task1_train_y(1:2900,:), opts);
    
    % SLeep to alter weights of neural network
    t_opts = struct;
    t_opts.t_ref        = 0.000;
    t_opts.threshold    =   1.0;
    t_opts.dt           = 0.001;
    t_opts.duration     = 0.035;
    t_opts.report_every = 0.001;
    t_opts.max_rate     =  inp_rate;

    
    sleep_opts.beta = [b1 b2 b3];
    sleep_opts.decay = 0.999;
    sleep_opts.W_inh = 0;
    sleep_opts.inc = inc;
    sleep_opts.dec = dec;
    sleep_opts.normW = 0;
    sleep_opts.DC = 0;
    [norm_nn, norm_constants] = normalize_nn_data(nn1, task1_train_x(1:2900,:));
    sleep_opts.alpha = norm_constants*alpha_scale;
    sleep_input = repmat(mean(task1_train_x), sleep_dur,1)'; 

    Snn = sleepnn_old(nn1, sleep_dur, t_opts, sleep_opts, ...
              sleep_input); % , threshold_scales
  
    % Train on task2
    opts.numepochs =  50;
    nn1.learningRate = task2_LR;
    control_nn2 = nntrain(control_nn1, task2_train_x(1:2900,:), task2_train_y(1:2900,:), opts);
    nn2 = nntrain(nn1, task2_train_x(1:2900,:), task2_train_y(1:2900,:), opts);
   
    % SLeep to alter weights of neural network
    t_opts = struct;
    t_opts.t_ref        = 0.000;
    t_opts.threshold    =   1.0;
    t_opts.dt           = 0.001;
    t_opts.duration     = 0.035;
    t_opts.report_every = 0.001;
    t_opts.max_rate     =  inp_rate;

    
    sleep_opts.beta = [b1 b2 b3];
    sleep_opts.decay = 0.999;
    sleep_opts.W_inh = 0;
    sleep_opts.inc = inc;
    sleep_opts.dec = dec;
    sleep_opts.normW = 0;
    sleep_opts.DC = 0;
    [norm_nn, norm_constants] = normalize_nn_data(nn2, train_x);
    sleep_opts.alpha = norm_constants*alpha_scale;
    sleep_input = repmat(mean(train_x), sleep_dur,1)'; 

    Snn = sleepnn_old(nn2, sleep_dur, t_opts, sleep_opts, ...
              sleep_input); % , threshold_scales
    
    [er,bad] = nntest(ideal_nn, task1_test_x, task1_test_y);
    accuracy(1,1,j) = 1-er;
    [er,bad] = nntest(control_nn2, task1_test_x, task1_test_y);
    accuracy(2,1,j) = 1-er;
    [er,bad] = nntest(Snn, task1_test_x, task1_test_y);
    accuracy(3,1,j) = 1-er;
    [er,bad] = nntest(ideal_nn, task2_test_x, task2_test_y);
    accuracy(1,2,j) = 1-er;
    [er,bad] = nntest(control_nn2, task2_test_x, task2_test_y);
    accuracy(2,2,j) = 1-er;
    [er,bad] = nntest(Snn, task2_test_x, task2_test_y);
    accuracy(3,2,j) = 1-er;
end
