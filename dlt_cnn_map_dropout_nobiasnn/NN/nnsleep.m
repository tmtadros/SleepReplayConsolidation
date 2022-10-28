function [nn, L]  = nnsleep(nn, train_x, sleep_opts)
%NNTRAIN runs sleep on a neural network
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');

numepochs = sleep_opts.numepochs;
tic;

[num_features,~] = size(train_x);
n = nn.n;
for i = 1:n
    nn.v{i} = zeros(nn.size(i),1);
end

nn.threshold = sleep_opts.beta;
nn.inc = sleep_opts.inc;
nn.dec = sleep_opts.dec;
max_rate = sleep_opts.max_rate;
nn.w_scale = sleep_opts.alpha;
dt = sleep_opts.dt;
nn.decay = sleep_opts.decay;
for i = 1 : numepochs 
    rescale_fac = 1/(dt*max_rate);
    spike_snapshot = rand(num_features,1) * rescale_fac/2;
    inp_image = spike_snapshot <= train_x(:,i);
    
    nn = nnff_sleep(nn, inp_image);
    nn = nnbp_sleep(nn);     
end
t = toc;

disp(['Sleep process took ' num2str(t) ' seconds']);
end

