function nn=sleepnn_old(nn, numiterations, opts, sleep_opts, sleep_input)
dt = opts.dt;
 
% Initialize network architecture
for l = 1 : numel(nn.size)
    blank_neurons = zeros(nn.size(l),1);
    nn.layers{l}.mem = blank_neurons;
    nn.layers{l}.refrac_end = blank_neurons;        
    nn.layers{l}.sum_spikes = blank_neurons;
    nn.layers{l}.total_spikes = zeros(numiterations, nn.size(l));
end
% inc = 0.01;
% dec = 0.001;

W_old = nn.W; 
 
num_features = size(sleep_input, 1);
sleep_opts.DC = 0;
% Time-stepped simulation
for t=1:numiterations
        % Create poisson distributed spikes from the input images
        %   (for all images in parallel)
        rescale_fac = 1/(dt*opts.max_rate);
        spike_snapshot = rand(num_features,1) * rescale_fac/2;
        inp_image = spike_snapshot <= sleep_input(:,t);
        
        nn.layers{1}.spikes = inp_image;
        nn.layers{1}.sum_spikes = nn.layers{1}.sum_spikes + inp_image;
        nn.layers{1}.total_spikes(t,:) = nn.layers{1}.spikes;
        
        %if mod(t, 100) == 1
        %    w1 = nn.W{1};
        %    w2 = nn.W{2};
        %    w3 = nn.W{3};
        %    save(strcat('/Users/timtadros/Documents/sleep-algorithm/mnist2/weights/wl1_iter_', num2str(t), '.mat'), 'w1')
        %    save(strcat('/Users/timtadros/Documents/sleep-algorithm/mnist2/weights/wl2_iter_', num2str(t), '.mat'), 'w2')
        %    save(strcat('/Users/timtadros/Documents/sleep-algorithm/mnist2/weights/wl3_iter_', num2str(t), '.mat'), 'w3')
        %end
        
        for l = 2 : numel(nn.size)
            % Get input impulse from incoming spike
            impulse = sleep_opts.alpha(l-1) * nn.layers{l-1}.spikes' * nn.W{l-1}';
            
            % Get input impulse from incoming spike
            impulse = impulse - sum(impulse)/length(impulse) * sleep_opts.W_inh;
            
            % Add input to membrane potential
            nn.layers{l}.mem = sleep_opts.decay*nn.layers{l}.mem + impulse';
            if l == 4
                nn.layers{l}.mem = nn.layers{l}.mem + sleep_opts.DC;
            end
            % Check for spiking
            nn.layers{l}.spikes = nn.layers{l}.mem >= opts.threshold ...
                 * (sleep_opts.beta(l-1)) ;

            % spikes_tmp = false(nn.size(l),1);
            % index = find(nn.layers{l}.mem >= opts.threshold * ...
            %              (sleep_opts.beta(l-1)));
            % prob = impulse(index)/max(impulse(index));
            % for ind = 1:length(index)
            %     spikes_tmp(index(ind)) = rand < (0.5 + prob(ind)/2);
            % end
            
            % nn.layers{l}.spikes = spikes_tmp;
            % nn.layers{l}.spikes = nn.layers{l}.mem >= opts.threshold ...
            %     * (sleep_opts.beta(l-1) + sleep_opts.theta * threshold_scales{l});
              
            % STDP
            post = nn.layers{l}.spikes;
            pre = nn.layers{l-1}.spikes;
            nn.W{l-1}(post==1,pre==1) = nn.W{l-1}(post==1,pre==1) + ...
                sleep_opts.inc * sigmoid(nn.W{l-1}(post==1,pre==1));
            nn.W{l-1}(post==1,pre==0) = nn.W{l-1}(post==1,pre==0) - ...
                sleep_opts.dec * sigmoid(nn.W{l-1}(post==1,pre==0));

            % Bound weights 
            % fi=find(nn.W{l-1}>W_old{l-1}*sleep_opts.delta_max);
            % nn.W{l-1}(fi)= W_old{l-1}(fi)*sleep_opts.delta_max; 
            % fi=find(nn.W{l-1}<W_old{l-1}*sleep_opts.delta_min);
            % nn.W{l-1}(fi)= W_old{l-1}(fi)*sleep_opts.delta_min; 
            
            % Reset
            nn.layers{l}.mem(nn.layers{l}.spikes) = 0;
             
            % Refractory period
            nn.layers{l}.refrac_end(nn.layers{l}.spikes) = t + opts.t_ref;
 
            % Store result for analysis later
            nn.layers{l}.sum_spikes = nn.layers{l}.sum_spikes + nn.layers{l}.spikes;            
            nn.layers{l}.total_spikes(t,:) = nn.layers{l}.spikes;
        end

end

% normalize change in weights
if sleep_opts.normW == 1
    for l = 2 : numel(nn.size)
        nn.W{l-1} = sleep_opts.gamma * nn.W{l-1}/(max(nn.W{l-1})-min(nn.W{l-1})) * ...
                    (max(W_old{l-1})-min(W_old{l-1}));
    end
end

end


function y = sigmoid(x)
    y = 2* (1.0 - (1.0 ./ (1 + exp(-x/0.001))));
end
