function [ X,y ] = create_permutations(width, overlap, num_images)
%Creates image quadrant patches with overlap based on parameter
% size = size of one side of square

X = zeros(num_images, width*width);

inds = 1:width*width;
shuffled_inds = inds(randperm(length(inds)));
overlapping_inds = shuffled_inds(1:overlap);
non_overlapping_inds = shuffled_inds(overlap+1:end);

non_overlapping_inds = non_overlapping_inds(randperm(length(non_overlapping_inds)));
num_indices = width*width/6-overlap;

figure;
for i = 1:num_images
    subplot(3,2,i)
    X(i,overlapping_inds) = 1;
    X(i,non_overlapping_inds((i-1)*num_indices+1:i*num_indices)) = 1;
    imshow(reshape(X(i,:), width, width));
end
y = zeros(num_images,num_images);
for i = 1:num_images
    y(i,i) = 1;
end
end

