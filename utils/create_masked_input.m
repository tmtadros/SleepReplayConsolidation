function [ sleep_x] = create_masked_input(X, numexamples, mask_size)
%Computes the average image and then returns masked versions of that input
sleep_input = mean(X).'; 
sleep_input = reshape(sleep_input, 28, 28);
sleep_x = zeros(numexamples, 28, 28);
for i = 1:numexamples
    x_pos = randi(28-mask_size);
    y_pos = randi(28-mask_size);
    sleep_x(i,x_pos:x_pos+mask_size, y_pos:y_pos + mask_size) = sleep_input(x_pos:x_pos+mask_size, y_pos:y_pos + mask_size);
end

sleep_x = reshape(sleep_x, numexamples, 784);


end

