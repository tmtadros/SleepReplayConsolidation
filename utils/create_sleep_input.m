function [ sleepX ] = create_sleep_input(X, y, i, numiterations)
%Creates input to sleep based on average by-digitimage
MNIST_average = zeros(784,i*2);
[~, labels] = max(y, [], 2);

for j = 1:i*2
    MNIST_average(:,j) = mean(X(find(labels == j), :)).';
end

sleepX = repmat(MNIST_average, 1, floor(numiterations/(i*2)));
indices = randperm(size(sleepX,2));
sleepX = sleepX(:,indices);

end

