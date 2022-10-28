function [ f ] = confmat_2tasks( nn, Snn, testX, testY)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[~, labels] = max(testY, [], 2);

awakeConfMat = zeros(4,4);
sleepConfMat = zeros(4,4);
for i = 1:4
    indices = find(labels == i);
    
    outputLabels = nnpredict(nn, testX(indices,:));
    sleepoutputLabels = nnpredict(Snn, testX(indices,:));

    for j = 1:4
        awakeConfMat(i,j) = sum(outputLabels == j);
        sleepConfMat(i,j) = sum(sleepoutputLabels == j);
    end
end
[x, y] = meshgrid(1:4);  % Create x and y coordinates for the strings

f = figure();
subplot(1,2,1)
imagesc(awakeConfMat); hold on;
colormap(flipud(gray));  % Change the colormap to gray (so higher values are
                         %   black and lower values are white)

textStrings = num2str(awakeConfMat(:), '%d');       % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
hStrings = text(x(:), y(:), textStrings(:), ...  % Plot the strings
                'HorizontalAlignment', 'center');
midValue = mean(get(gca, 'CLim'));  % Get the middle value of the color range

textColors = repmat(awakeConfMat(:) > midValue, 1, 3);  % Choose white or black for the
                                               %   text color of the strings so
                                               %   they can be easily seen over
                                               %   the background color
set(hStrings, {'Color'}, num2cell(textColors, 2));  % Change the text colors

set(gca, 'XTick', 1:4, ...                             % Change the axes tick marks
         'XTickLabel', {'0', '1', '2', '3'}, ...  %   and tick labels
         'YTick', 1:4, ...
         'YTickLabel', {'0', '1', '2', '3'}, ...
         'TickLength', [0 0]);
for i = 1:4
   plot([.5,21.5],[i-.5,i-.5],'k-');
   plot([i-.5,i-.5],[.5,21.5],'k-');
end
xlabel('Prediction')
ylabel('Actual')
title('Before sleep')

subplot(1,2,2)
imagesc(sleepConfMat); hold on;

colormap(flipud(gray));  % Change the colormap to gray (so higher values are
                         %   black and lower values are white)

textStrings = num2str(sleepConfMat(:), '%d');       % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
hStrings = text(x(:), y(:), textStrings(:), ...  % Plot the strings
                'HorizontalAlignment', 'center');
midValue = mean(get(gca, 'CLim'));  % Get the middle value of the color range

textColors = repmat(sleepConfMat(:) > midValue, 1, 3);  % Choose white or black for the
                                               %   text color of the strings so
                                               %   they can be easily seen over
                                               %   the background color
set(hStrings, {'Color'}, num2cell(textColors, 2));  % Change the text colors

set(gca, 'XTick', 1:4, ...                             % Change the axes tick marks
         'XTickLabel', {'0', '1', '2', '3'}, ...  %   and tick labels
         'YTick', 1:4, ...
         'YTickLabel', {'0', '1', '2', '3'}, ...
         'TickLength', [0 0]);
for i = 1:4
   plot([.5,21.5],[i-.5,i-.5],'k-');
   plot([i-.5,i-.5],[.5,21.5],'k-');
end
xlabel('Prediction')
ylabel('Actual')
title('After sleep')

end