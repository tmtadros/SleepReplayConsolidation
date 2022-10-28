function [ f ] = confmat( nn, Snn, testX, testY)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[~, labels] = max(testY, [], 2);

awakeConfMat = zeros(10,10);
sleepConfMat = zeros(10,10);
for i = 1:10
    indices = find(labels == i);
    
    outputLabels = nnpredict(nn, testX(indices,:));
    sleepoutputLabels = nnpredict(Snn, testX(indices,:));

    for j = 1:10
        awakeConfMat(i,j) = sum(outputLabels == j);
        sleepConfMat(i,j) = sum(sleepoutputLabels == j);
    end
end
[x, y] = meshgrid(1:10);  % Create x and y coordinates for the strings

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

set(gca, 'XTick', 1:10, ...                             % Change the axes tick marks
         'XTickLabel', {'0', '1', '2', '3', '4', '5','6', '7', '8', '9'}, ...  %   and tick labels
         'YTick', 1:10, ...
         'YTickLabel', {'0', '1', '2', '3', '4', '5','6', '7', '8', '9'}, ...
         'TickLength', [0 0]);
for i = 1:10
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

set(gca, 'XTick', 1:10, ...                             % Change the axes tick marks
         'XTickLabel', {'0', '1', '2', '3', '4', '5','6', '7', '8', '9'}, ...  %   and tick labels
         'YTick', 1:10, ...
         'YTickLabel', {'0', '1', '2', '3', '4', '5','6', '7', '8', '9'}, ...
         'TickLength', [0 0]);
for i = 1:10
   plot([.5,21.5],[i-.5,i-.5],'k-');
   plot([i-.5,i-.5],[.5,21.5],'k-');
end
xlabel('Prediction')
ylabel('Actual')
title('After sleep')

end

