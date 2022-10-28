function [ sleep_x] = create_masked_input_CIFAR(X, numexamples, mask_size)
%Computes the average image and then returns masked versions of that input
sleep_input = mean(X).'; 
sleep_x = zeros(numexamples, 2048);
for i = 1:numexamples
    x_pos = randi(2048-mask_size);
    y_pos = randi(2048-mask_size);
    sleep_x(i,x_pos:x_pos+mask_size) = sleep_input(x_pos:x_pos+mask_size);
end

sleep_x = reshape(sleep_x, numexamples, 2048);


end
