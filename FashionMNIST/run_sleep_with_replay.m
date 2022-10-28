function [ accuracy ] = run_sleep_with_replay(task_order, percent_replay, do_sleep)
load fashionmnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

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

if percent_replay < 0.025
    x = [239.515363 0.032064 0.003344 14.548273 44.560317 38.046326 55.882454];
elseif percent_replay >= 0.1
    x = [37.141606 0.100000 0.000001 37.315671 0.318765 0.000000 0.100000 ];
elseif percent_replay == 0.05
    x = [21.000000 0.100000 0.000001 0.862173 0.318765 0.000000 0.100000];   
elseif percent_replay == 0.025 
    x = [142.445655 0.003775 0.003577 26.790053 19.800337 25.706878 43.382730];
end
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

for ii = 1:5
    i=task_order(ii);
    indices = find(tasks == i);
    if ii <= 1
        nn1 = nntrain(nn_pointer, train_x(indices(1:5000),:), train_y(indices(1:5000),:), opts);
    else
        inds2 = find(ismember(tasks, task_order(1:ii-1)));
        num_examples = length(inds2)
        num_replayed_examples = round(num_examples*percent_replay,-2)
        inds2 = inds2(randperm(length(inds2)));
        
        tr_x = vertcat(train_x(indices(1:5000),:), train_x(inds2(1:num_replayed_examples),:));
        tr_y = vertcat(train_y(indices(1:5000),:), train_y(inds2(1:num_replayed_examples),:));
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
    if do_sleep == 1
        Snn = sleepnn_old(nn_pointer, sleep_period, t_opts, sleep_opts, ...
                  sleep_input'); % , threshold_scales

        nn_pointer = Snn;
        for k = 1:length(layers)
            Snn.W{layers(k)} = nn1.W{layers(k)};  
        end
    else
        Snn = nn_pointer;
    end
    NNs{ii} = {nn1, Snn};
end

er = nntest(Snn, test_x, test_y);
accuracy = (1-er)*100;
end

