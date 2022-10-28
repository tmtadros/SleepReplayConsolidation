function [task_assignment, newY] = create_domain_task(X,y)
%creates the domain task in the Gido paper: 
% 5 tasks representing digits 0-1, 2-3, 4-5, 6-7, 8-9
% and the label says if its an even or odd digit

num_images = size(X,1);

task_assignment = zeros(num_images,1);
newY = zeros(num_images, 2);


[~, labels] = max(y, [], 2);

for i = 1:num_images
    if labels(i) == 0 || labels(i) == 1
        task_assignment(i) = 1;
    elseif labels(i) == 2 || labels(i) == 3
        task_assignment(i) = 2;
    elseif labels(i) == 4 || labels(i) == 5
        task_assignment(i) = 3;
    elseif labels(i) == 6 || labels(i) == 7
        task_assignment(i) = 4;
    elseif labels(i) == 8 || labels(i) == 9
        task_assignment(i) = 5;
    end
    
    if rem(labels(i), 2) == 0
        newY(i,2) = 1;
    else
        newY(i,1) = 1;
    end
end

