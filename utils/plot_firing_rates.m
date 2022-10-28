function [ f1,f2,f3, f4 ] = plot_firing_rates(Snn)
% Plots firing rates during sleep

% Input layer
input_spikes_sum = Snn.layers{1}.sum_spikes;
%input_spikes_all = Snn.layers{1}.total_spikes;
f1 = figure();
imagesc(reshape(input_spikes_sum, 28,28)');
title('Sleep input')

% Intermediate layer #1
int2_spikes = Snn.layers{2}.sum_spikes;
f2 = figure();
histogram(int2_spikes);
title('Layer 2 firing rate histogram')

% Intermediate layer #2
int3_spikes = Snn.layers{3}.sum_spikes;
f3 = figure();
histogram(int3_spikes);
title('Layer 3 firing rate histogram')

% Output layer
f4 = figure();
output_spikes_sum = Snn.layers{4}.sum_spikes;
bar(output_spikes_sum);
xlabel('Output Neuron')
ylabel('Total number of spikes')
title('Spike count of output neurons')
end

