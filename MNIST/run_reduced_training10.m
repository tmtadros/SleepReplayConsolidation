function [ performance ] = run_reduced_training10( do_sleep )

x = [42.162268 0.022175 0.076886 37.312702 7.101362 32.780841 14.308178];

inp_rate=x(1);
inc=x(2); 
dec=x(3);
b1=x(4); % 1.1
b2 =x(5);
b3 = x(6);
alpha_scale = x(7);  % 4.0

addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));

%% load mnist_uint8 and create task;
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%Initialize net
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
opts.numepochs =  10;
opts.batchsize = 100;


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
%%
gamma = 1.0;
lr=0.01;
nn_curr = nn;
nn_curr.gamma = gamma;
nn_curr.learningRate = lr;
opts.numepochs=10;
num_images = 5000; 

nn_curr = nntrain(nn_curr, train_x(1:num_images,:), train_y(1:num_images,:), opts);
[er,bad] = nntest(nn_curr, test_x, test_y);

sleep_period = 20000;
sleep_input = create_masked_input(train_x, sleep_period, 10);
[~, norm_constants] = normalize_nn_data(nn_curr, train_x);
sleep_opts.alpha = norm_constants*alpha_scale; %*2.25;
tic
if do_sleep == 1
    % Run NREM
    Snn = sleepnn_old(nn_curr, sleep_period, t_opts, sleep_opts, ...
              sleep_input'); % , threshold_scales
else
    Snn = nn_curr;
end
[er,bad] = nntest(Snn, train_x, train_y);

performance = (1-er)*100;

end