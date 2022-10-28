function [ accuracy1,accuracy2 ] = run_icarl_fashionmnist(task_order, do_sleep, memory, epochs, distillation, herding)

%% load mnist_uint8 and create task;
load fashionmnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

[tasks, train_y] = create_class_task(train_x, train_y);
[test_tasks, test_y] = create_class_task(test_x, test_y);
[~,tr_labels] = max(train_y, [], 2);


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
nn.output ='softmax';
nn.learningRate = 0.065;
nn.momentum = 0.5;
nn.dropoutFraction = 0.2;
nn.learn_bias = 0;
opts.numepochs =  epochs;
opts.batchsize = 100;

if memory == 50
    x = [142.945655 0.003775 0.009083 39.917543 39.468197 12.188520 5.573700];
elseif memory == 100 || memory == 200 || memory == 500
    x = [42.162268 0.060285 0.003577 36.717054 20.800337 30.249532 12.253680];
elseif memory == 2000 || memory == 1000 || memory == 5000
    x = [20.000000 0.100000 0.000001 0.862173 0.318765 0.000000 0.100000];
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

task_order = task_order; 
%classes = 1:10;

numiterations = 15000;
%task_order = [1,2,3,4,5];
%classes = [1,2,3,4,5,6,7,8,9,10];
Xsets = {};
classes = convert_task_to_labels(task_order);

if memory == 20000
    K  = 10000;
else
    K = memory;
end
for ii = 1:5
    i=task_order(ii);
    indices = find(tasks == i);
    
    if ii <= 1 % train first task and add to exemplar sets
        nn1 = nntrain(nn_pointer, train_x(indices(1:5000),:), train_y(indices(1:5000),:), opts);
	if herding == 0
            Xsets{2*ii-1} = construct_exemplar_set2(train_x(tr_labels==i*2-1,:), round(K/2), nn1);
            Xsets{2*ii} = construct_exemplar_set2(train_x(tr_labels==i*2,:), round(K/2), nn1);
	elseif herding == 1
            Xsets{2*ii-1} = construct_exemplar_set_random(train_x(tr_labels==i*2-1,:), round(K/2), nn1);
            Xsets{2*ii} = construct_exemplar_set_random(train_x(tr_labels==i*2,:), round(K/2), nn1);		
	end
	if memory == 20000
            K = 20000;
        end
    else
        % concatenate old dataset to new dataset
        tr_x = train_x(indices(1:5000),:);
        tr_y = train_y(indices(1:5000),:);
        
        for j = 1:(ii-1)*2
           new_label = classes(j);
           y = zeros(size(Xsets{j},1), 10);
           y(:,new_label) = 1;
           if distillation == 1
               nn_temp = nnff(nn_pointer, Xsets{j}, y);
               y = nn_temp.a{4};
           end
           %temp_nn = nnff(nn_pointer, Xsets{j}, y);
           %y(:,new_label) = 1;
           %y = temp_nn.a{length(temp_nn.a)};
           tr_x = vertcat(tr_x, Xsets{j});
           tr_y = vertcat(tr_y,y); 
           
        end
        if rem(length(tr_x), 100) ~= 0
            tr_x = vertcat(tr_x, train_x(indices(5000:5000+99-rem(length(tr_x),100)),:));
            tr_y = vertcat(tr_y, train_y(indices(5000:5000+99-rem(length(tr_y),100)),:));
        end

            
        nn1 = nntrain(nn_pointer, tr_x, tr_y, opts);
	if herding == 0
            Xsets{2*ii-1} = construct_exemplar_set2(train_x(tr_labels==(i-1)*2+1,:), round(K/(ii*2)), nn1);
            Xsets{2*ii} = construct_exemplar_set2(train_x(tr_labels==(i-1)*2+2,:), round(K/(ii*2)), nn1);
	elseif herding == 1
            Xsets{2*ii-1} = construct_exemplar_set_random(train_x(tr_labels==(i-1)*2+1,:), round(K/(ii*2)), nn1);
            Xsets{2*ii} = construct_exemplar_set_random(train_x(tr_labels==(i-1)*2+2,:), round(K/(ii*2)), nn1);
	end	    
        for j = 1:(ii-1)*2
            Xsets{j} = reduce_example_set(Xsets{j}, round(K/(ii*2)));
        end
    end
    
    % Train on taski
    nn_pointer = nn1;
    
    
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
    else
        Snn = nn_pointer;
    end

    nn_pointer = Snn;

    NNs{ii} = {nn1, Snn};
    %confFig = confmat(nn1, Snn, test_x, test_y);
    %[f1, f2, f3, f4] = plot_firing_rates(Snn);
    %[w1, w2] = plot_weight_distributions(nn1, Snn);    
end


if memory > 0
example_x = zeros(K, nn.size(1));
example_y = zeros(K, nn.size(4));
for i = 1:length(Xsets)
    class = classes(i);
    example_x((i-1)*K/10+1:i*K/10,:) = Xsets{i};
    example_y((i-1)*K/10+1:i*K/10,class) = 1;
end
end
er1 = nntest(Snn, test_x, test_y);
if memory > 0
    er2 = nntest_iCarl(Snn, test_x, test_y, example_x, example_y);
else
    er2 = 1;
end
accuracy1 = (1-er1)*100;
accuracy2 = (1-er2)*100;
end
