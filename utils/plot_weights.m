function [ w_output_before, w_output_after ] = plot_weights(nn, Snn)
% plots sum of weights to each output neuron
w_before = nn.W{3};
w_sleep = Snn.W{3};

w_output_before = sum(w_before, 2);
w_output_after = sum(w_sleep, 2);

figure(1);
for i = 1:10
    subplot(5,2,i);
    scatter(w_before(i,:), w_sleep(i,:));
    xlabel("Before")
    ylabel("After")
end

