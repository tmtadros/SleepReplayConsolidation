function [ classes ] = convert_task_to_labels( task_order )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

classes = [];
i = 1;
for j = 1:length(task_order)
     classes(i) = task_order(j)*2-1;
     classes(i+1) = task_order(j)*2;
     i = i + 2;
end

end

