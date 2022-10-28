function [ accuracy1,accuracy2 ] = run_sleep_with_variable_task2_length(task_order, task2_num_epochs)
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

rng('default');
rng(1000);

[tasks, train_y] = create_class_task(train_x, train_y);
[test_tasks, test_y] = create_class_task(test_x, test_y);

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


x = [239.515363 0.032064 0.003344 14.548273 44.560317 38.046326 55.882454];
inp_rate=x(1);
inc=x(2); 
dec=x(3);
b1=x(4); % 1.1
b2 =x(5);
b3 = x(6);
alpha_scale = x(7);

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
accuracy = zeros(10,6);
plot_i = 1;
numiterations = 15000;
layers = [];

for ii = 1:2
    i=task_order(ii);
    indices = find(tasks == i);
    if ii <= 1
        nn1 = nntrain(nn_pointer, train_x(indices(1:5000),:), train_y(indices(1:5000),:), opts);
    else
        opts.numepochs =  task2_num_epochs;
        
        tr_x = train_x(indices(1:5000),:);
        tr_y = train_y(indices(1:5000),:);
        nn1 = nntrain(nn_pointer, tr_x, tr_y, opts);
    end
    % Train on taski
%    nn1 = nntrain(nn_pointer, train_x(indices(1:5000),:), train_y(indices(1:5000),:), opts);
    nn_pointer = nn1;
    
    labels{plot_i} = strcat('Train Task: ', num2str(i));
    
    % sleep
    indices = find(ismember(tasks, task_order(1:ii)));
    sleep_period = numiterations +(ii-1)*numiterations/3;
    sleep_input = create_masked_input(train_x(indices,:), sleep_period, 10);
    [~, norm_constants] = normalize_nn_data(nn_pointer, train_x(indices,:));
    sleep_opts.alpha = norm_constants*alpha_scale; %*2.25;

    % Run NREM
    Snn = sleepnn_old(nn_pointer, sleep_period, t_opts, sleep_opts, ...
                  sleep_input'); % , threshold_scales

    nn_pointer = Snn;

    NNs{ii} = {nn1, Snn};
end

er = nntest(Snn, test_x(test_tasks==1,:), test_y(test_tasks==1,:));
accuracy1 = (1-er)*100
er = nntest(Snn, test_x(test_tasks==2,:), test_y(test_tasks==2,:));
accuracy2 = (1-er)*100
end