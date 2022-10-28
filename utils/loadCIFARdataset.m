function [ train_x, train_y, test_x, test_y ] = loadCIFARdataset( )
%
train_x = zeros(50000, 3072);
train_y = zeros(50000,1);
for i = 1:5
    load(strcat('data_batch_', num2str(i)));
    train_x((i-1)*10000+1:i*10000,:) = double(data)/255;
    train_y((i-1)*10000+1:i*10000) = labels;
end
load test_batch;
test_x = double(data)/255;
test_y = labels;

% y is a vector of labels
train_y_oh = zeros( size( train_y, 1 ), 10 );
test_y_oh = zeros( size( test_y, 1 ), 10);
% assuming class labels start from one
for i = 1:10
    train_rows = train_y == i-1;
    test_rows = test_y == i-1;
    train_y_oh( train_rows, i ) = 1;
    test_y_oh( test_rows,i) = 1;
end

train_y = train_y_oh;
test_y = test_y_oh;


end

