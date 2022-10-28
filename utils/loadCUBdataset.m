function [ train_x, train_y, test_x, test_y ] = loadCUBdataset()

train_set = load('cub200_resnet50_train.mat');
test_set = load('cub200_resnet50_test.mat');

train_x = train_set.X;
train_y = squeeze(train_set.y);
test_x = test_set.X;
test_y = squeeze(test_set.y);

%% One hot encoding
% y is a vector of labels
train_y_oh = zeros( size( train_y, 1 ), 200 );
test_y_oh = zeros( size( test_y, 1 ), 200);
% assuming class labels start from one
for i = 1:200
    train_rows = train_y == i;
    test_rows = test_y == i;
    train_y_oh( train_rows, i ) = 1;
    test_y_oh( test_rows,i) = 1;
end

train_y =train_y_oh;
test_y = test_y_oh;
end

