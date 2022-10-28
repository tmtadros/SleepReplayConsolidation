function [f1, f2] = plot_weight_distributions(Ann, Snn)
%plots distribution of weights for each layer
f1 = figure();
ind = 1;
for i = 1:3
    awake_weights = Ann.W{i};
    sleep_weights = Snn.W{i};
    subplot(3,2,ind);
    histogram(awake_weights(:),'facealpha',.5); hold on
    histogram(sleep_weights(:),'facealpha',.5);
    ylim([-50 inf])
    title('Layer ' + num2str(i));
    legend({'Before', 'After sleep'})
    subplot(3,2,ind+1)
    histogram(sleep_weights(:) - awake_weights(:),'facealpha',.5);
    ylim([-50 inf])
    title('Weight differences')
    ind = ind + 2;
end

% Plot distributions based on increasing and decreasing weights
f2 = figure();
ind = 1;
for i = 1:3
    awake_weights = Ann.W{i}(:);
    sleep_weights = Snn.W{i}(:);
    w_diffs = sleep_weights - awake_weights;
    increasing = find(w_diffs > 0);
    decreasing = find(w_diffs < 0);

    subplot(3,2,ind)
    histogram(awake_weights(increasing),'facealpha',.5); hold on
    histogram(awake_weights(decreasing),'facealpha',.5);
    ylim([-50 inf])

    title('Before, Layer ' + num2str(i));
    legend({'Increasing', 'Decreasing'})
    subplot(3,2,ind+1)
    histogram(sleep_weights(increasing),'facealpha',.5); hold on
    histogram(sleep_weights(decreasing),'facealpha',.5);
    ylim([-50 inf])
    title('After Sleep, Layer ' + num2str(i));
    legend({'Increasing', 'Decreasing'})
    ind = ind + 2;
end

