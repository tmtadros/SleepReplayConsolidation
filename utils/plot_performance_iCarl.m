function [ f ] = plot_performance_iCarl(NNs, labels, X, Y, S, Sy, task_labels, task)
%Plots the performance for each task seperately before and after sleep
%(tasks seen so far)

numNNs = length(NNs);
num_tasks = length(unique(task_labels));

performance = zeros(length(labels), num_tasks);

plot_i = 1;
sleep_indices = [];
for i = 1:numNNs
    if length(NNs{i}) > 1
        nn1 = NNs{i}{1};
        Snn = NNs{i}{2};
        for j = 1:num_tasks
            indices = find(task_labels == j);
            [er, ~] = nntest_iCarl(nn1, X(indices,:), Y(indices,:), S, Sy);
            performance(plot_i,j) = (1-er)*100;
            [er, ~] = nntest_iCarl(Snn, X(indices,:), Y(indices,:),S,Sy);
            performance(plot_i+1,j) = (1-er)*100;
        end
        indices = find(task_labels <= task);
        [er, ~] = nntest_iCarl(nn1, X(indices,:), Y(indices,:),S,Sy);
        performance(plot_i,num_tasks+1) = (1-er)*100;
        [er, ~] = nntest_iCarl(Snn, X(indices,:), Y(indices,:),S,Sy);
        performance(plot_i+1,num_tasks+1) = (1-er)*100;
        sleep_indices(end+ 1) = plot_i  + 1;
        plot_i = plot_i + 2;
    else    
        nn1 = NNs{i}{1};
        for j = 1:num_tasks
            indices = find(task_labels == j);
            [er, ~] = nntest_iCarl(nn1, X(indices,:), Y(indices,:),S,Sy);
            performance(plot_i,j) = (1-er)*100;
        end
        indices = find(task_labels <= task);
        [er, ~] = nntest_iCarl(nn1, X(indices,:), Y(indices,:),S,Sy);
        performance(plot_i,num_tasks+1) = (1-er)*100;
        plot_i = plot_i + 1;
    end
end

legend_labels = {};
for i = 1:length(labels)
    data = labels{i}
    if ~isempty(data)
        legend_labels{end+1} = data(1);
    end
end

barlabels = {'Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'All tasks'};
f = figure();
imagesc(performance'); hold on;
x = 1:10';
y = 1:6;
[X,Y] = meshgrid(x,y);
a = performance';
textStrings = num2str(a(:), '%0.2f');       % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
hStrings = text(X(:), Y(:), textStrings(:), ...  % Plot the strings
                'HorizontalAlignment', 'center');
midValue = mean(get(gca, 'CLim'));  % Get the middle value of the color range
%colormap('jet')
textColors = repmat(a(:) < midValue, 1, 3);  % Choose white or black for the
                                               %   text color of the strings so
                                               %   they can be easily seen over
                                               %   the background color
set(hStrings, {'Color'}, num2cell(textColors, 2));  % Change the text colors

set(gca, 'XTick', 1:10, ...                             % Change the axes tick marks
         'XTickLabel', legend_labels, ...  %   and tick labels
         'YTick', 1:6, ...
         'YTickLabel', barlabels, ...
         'TickLength', [0 0]);
for i = 1:10
   plot([.5,21.5],[i-.5,i-.5],'k-');
   plot([i-.5,i-.5],[.5,21.5],'k-');
end
xlabel('Phase')
ylabel('Accuracy')


barlabels = {'Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Average'};
f = figure();
H = bar(performance');
set(H(sleep_indices), 'FaceColor', 'r')
set(gca, 'XTickLabel', barlabels)
legend(cellstr(legend_labels))

end
