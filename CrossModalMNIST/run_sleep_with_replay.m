function [ accuracy ] = run_sleep_with_replay(task_order, percent_replay, do_sleep)
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


if percent_replay == 0.0
    x = [226.894481 0.053742 0.005592 4.649797 9.239378 7.213289 9.301153 14084.545382 7718.641016];
elseif percent_replay == 0.01
    x = [36.472339 0.048044 0.042630 38.862027 5.929705 26.392342 6.717796 209.495050 4170.928949];
elseif percent_replay == 0.025
    x = [75.249380 0.043078 0.015357 42.704997 36.456573 37.711485 2.136156 4144.432204 11856.030957];
elseif percent_replay == 0.05
    x = [131.962075 0.058340 0.057663 43.537606 11.523920 18.678190 1.544190 1920.118349 1830.165840];
else
    x = [37.722339 0.042346 0.027100 46.560069 8.310178 34.473181 0.569274 8825.575973 8812.494927];
end

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

%% Train first task
nn1 = nntrain(nn_pointer, train_x1, train_y1, opts);
nn_pointer = nn1;
if do_sleep == 1
    % sleep
    sleep_period = uint8(l1);
    sleep_input = create_masked_input(train_x1, sleep_period, 10);
    [~, norm_constants] = normalize_nn_data(nn_pointer, train_x1);
    sleep_opts.alpha = norm_constants*alpha_scale; %*2.25;

    % Run NREM
    Snn = sleepnn_old(nn_pointer, sleep_period, t_opts, sleep_opts, ...
              sleep_input'); % , threshold_scales
else
    Snn = nn_pointer;
end

inds2 = 1:length(train_x2);
num_replayed_examples = round(length(train_x2)*percent_replay,-2)
inds2 = inds2(randperm(length(inds2)));
tr_x = vertcat(train_x2, train_x1(inds2(1:num_replayed_examples),:));
tr_y = vertcat(train_y2, train_y1(inds2(1:num_replayed_examples),:));
nn1 = nntrain(nn_pointer, tr_x, tr_y, opts);
nn_pointer = nn1;
    
if do_sleep == 1    
    % sleep    
    training_data = train_x1;   
    sleep_period = uint8(l2);
    sleep_input = create_masked_input(training_data, sleep_period, 10);
    [~, norm_constants] = normalize_nn_data(nn_pointer, training_data);
    sleep_opts.alpha = norm_constants*alpha_scale; %*2.25;

    % Run NREM
    Snn = sleepnn_old(nn_pointer, sleep_period, t_opts, sleep_opts, ...
              sleep_input'); % , threshold_scales

    nn_pointer = Snn;
else
    Snn = nn_pointer;
end

test_x = vertcat(test_x1, test_x2);
test_y = vertcat(test_y1, test_y2);
er = nntest(Snn, test_x, test_y);
accuracy = (1-er)*100;
end