function [awake_correct, sleep_correct ] = change_in_performance(awakeNN, sleepNN, test_x, test_y)
% Computes number of images of each class classified correctly by each
% network

[vals, labels] = max(test_y'); 
awake_correct = zeros(10,1);
sleep_correct = zeros(10,1);
for i = 1:10
    indices = labels == i;
    [er, bad] = nntest(awakeNN, test_x(indices,:), test_y(indices,:));

    awake_correct(i) = (1-er)*100;
    [er, bad] = nntest(sleepNN, test_x(indices,:), test_y(indices,:));
    sleep_correct(i) = (1-er)*100;
end

figure()
subplot(121)
bar(awake_correct)
subplot(122)
bar(sleep_correct-awake_correct)
end

