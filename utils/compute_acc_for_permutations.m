function [performance,end_accuracy]=compute_acc_for_permutations(NNs, lasttask, labels, X, Y, task_labels)

numNNs = length(NNs);
num_tasks = length(unique(task_labels));

performance = zeros(length(labels), num_tasks+1);

plot_i = 1;
sleep_indices = [];
for i = 1:numNNs
    if length(NNs{i}) > 1
        nn1 = NNs{i}{1};
        Snn = NNs{i}{2};
        for j = 1:num_tasks
            indices = find(task_labels == j);
            [er, ~] = nntest(nn1, X(indices,:), Y(indices,:));
            performance(plot_i,j) = (1-er)*100;
            [er, ~] = nntest(Snn, X(indices,:), Y(indices,:));
            performance(plot_i+1,j) = (1-er)*100;
        end
        [er, ~] = nntest(nn1, X, Y);
        performance(plot_i,num_tasks+1) = (1-er)*100;
        [er, ~] = nntest(Snn, X, Y);
        performance(plot_i+1,num_tasks+1) = (1-er)*100;
        sleep_indices(end+ 1) = plot_i  + 1;
        plot_i = plot_i + 2;
    else    
        nn1 = NNs{i};
        for j = 1:num_tasks
            indices = find(task_labels == j);
            [er, ~] = nntest(nn1, X(indices,:), Y(indices,:));
            performance(plot_i,j) = (1-er)*100;
        end
        indices = find(task_labels <= task);
        [er, ~] = nntest(nn1, X(indices,:), Y(indices,:));
        performance(plot_i,num_tasks+1) = (1-er)*100;
        plot_i = plot_i + 1;
    end
end
Snn = NNs{lasttask}{2};
[er, ~] = nntest(Snn, X, Y);
end_accuracy =(1-er)*100;
end
