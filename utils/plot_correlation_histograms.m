function [ f ] = plot_correlation_histograms( cbefore, cafter )

[num_layers, ~, ~] = size(cbefore);

f = figure;
for i = 2:num_layers
    subplot(1,num_layers-1, i-1);
    ca = squeeze(cafter(i,:,:));
    cb = squeeze(cbefore(i,:,:));
    histogram(diag(ca) - diag(cb), 10); hold on;

    ca_off = ca;
    ca_off(logical(eye(size(ca)))) = []; 
    ca_off = reshape(ca_off,9,10);  % Or 3,4 or whatever.
    cb_off = cb;
    cb_off(logical(eye(size(cb)))) = []; 
    cb_off = reshape(cb_off,9,10);  % Or 3,4 or whatever.
    histogram(ca_off(:) - cb_off(:), 10);
    title(['Layer ' num2str(i)])
    legend('Same class', 'Different class')
    xlabel('Correlation change')
    ylabel('N')
end

