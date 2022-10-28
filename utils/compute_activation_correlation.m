function [ correlations_before, correlations_after ] = compute_activation_correlation(nn1, Snn, test_x, test_y)
%Computes correlation of activations of each layer before and after sleep

[~, labels] = max(test_y, [], 2);
nn_before = nnff(nn1, test_x, test_y);
nn_after = nnff(Snn, test_x, test_y);
correlations_before = zeros(3, 10, 10);
correlations_after = zeros(3, 10, 10);

for i = 2:nn1.n
    activations = nn_before.a{i};
    Sactivations = nn_after.a{i};

    correlations = corrcoef(activations');
    Scorrelations = corrcoef(Sactivations');
    for j = 1:10
        for k = j:10
            rel_corrs = correlations(labels==j, labels==k);
            Srel_corrs = Scorrelations(labels==j, labels==k);

            correlations_before(i,j,k) = nanmean(rel_corrs(:));
            correlations_before(i,k,j) = nanmean(rel_corrs(:));
            correlations_after(i,j,k) = nanmean(Srel_corrs(:));
            correlations_after(i,k,j) = nanmean(Srel_corrs(:));
        end
    end
end


for i = 2:nn1.n
    Cbefore = squeeze(correlations_before(i,:,:));
    Cafter = squeeze(correlations_after(i,:,:));
    
    figure();
    [x, y] = meshgrid(1:10); 
    subplot(1,2,1)
    imagesc(Cbefore); hold on;
    colormap(flipud(gray));  % Change the colormap to gray (so higher values are
                             %   black and lower values are white)
    caxis([0 max(Cbefore(:))])

    textStrings = num2str(Cbefore(:), '%0.2f');       % Create strings from the matrix values
    textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
    hStrings = text(x(:), y(:), textStrings(:), ...  % Plot the strings
                    'HorizontalAlignment', 'center');
    midValue = mean(get(gca, 'CLim'));  % Get the middle value of the color range

    textColors = repmat(Cbefore(:) > midValue, 1, 3);  % Choose white or black for the
                                                   %   text color of the strings so
                                                   %   they can be easily seen over
                                                   %   the background color
    set(hStrings, {'Color'}, num2cell(textColors, 2));  % Change the text colors

    set(gca, 'XTick', 1:10, ...                             % Change the axes tick marks
             'XTickLabel', {'0', '1', '2', '3', '4', '5','6', '7', '8', '9'}, ...  %   and tick labels
             'YTick', 1:10, ...
             'YTickLabel', {'0', '1', '2', '3', '4', '5','6', '7', '8', '9'}, ...
             'TickLength', [0 0]);
    for j = 1:10
       plot([.5,21.5],[j-.5,j-.5],'k-');
       plot([j-.5,j-.5],[.5,21.5],'k-');
    end
    title('Before sleep')

    subplot(1,2,2)
    imagesc(Cafter); hold on;

    colormap(flipud(gray));  % Change the colormap to gray (so higher values are
                             %   black and lower values are white)
    caxis([0 max(Cbefore(:))])

    textStrings = num2str(Cafter(:), '%0.2f');       % Create strings from the matrix values
    textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
    hStrings = text(x(:), y(:), textStrings(:), ...  % Plot the strings
                    'HorizontalAlignment', 'center');
    midValue = mean(get(gca, 'CLim'));  % Get the middle value of the color range

    textColors = repmat(Cafter(:) > midValue, 1, 3);  % Choose white or black for the
                                                   %   text color of the strings so
                                                   %   they can be easily seen over
                                                   %   the background color
    set(hStrings, {'Color'}, num2cell(textColors, 2));  % Change the text colors

    set(gca, 'XTick', 1:10, ...                             % Change the axes tick marks
             'XTickLabel', {'0', '1', '2', '3', '4', '5','6', '7', '8', '9'}, ...  %   and tick labels
             'YTick', 1:10, ...
             'YTickLabel', {'0', '1', '2', '3', '4', '5','6', '7', '8', '9'}, ...
             'TickLength', [0 0]);
    for j = 1:10
       plot([.5,21.5],[j-.5,j-.5],'k-');
       plot([j-.5,j-.5],[.5,21.5],'k-');
    end
    title('After sleep')
    suptitle(strcat('Activation correlations in layer ', num2str(i)))
end
%hp4 = get(subplot(2,2,4),'Position');
%colorbar('Position', [hp4(1)+hp4(3)+0.01  hp4(2)  0.1  hp4(2)+hp4(3)*2.1])
end


