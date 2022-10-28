function [task_assignment, newY] = create_class_task_CUB(X,y)
%creates the class task in the Gido paper: 
% 5 tasks representing digits 0-1, 2-3, 4-5, 6-7, 8-9
% and the label says which digit it is

num_images = size(X,1);

task_assignment = zeros(num_images,1);
newY = y;

[~, labels] = max(y, [], 2);
num_classes = max(labels);
for i = 1:num_images
    if labels(i) <= num_classes/5
        task_assignment(i) = 1;
    elseif labels(i) <= 2*num_classes/5
        task_assignment(i) = 2;
    elseif labels(i) <= 3*num_classes/5
        task_assignment(i) = 3;
    elseif labels(i) <= 4*num_classes/5
        task_assignment(i) = 4;
    elseif labels(i) <= num_classes
        task_assignment(i) = 5;
    end
end