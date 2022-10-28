function [ sleep_x] = create_masked_input_CIFAR_specific(X, numexamples, mask_size)
%Computes the average image and then returns masked versions of that input
sleep_x = zeros(numexamples, 2048);
for i = 1:numexamples
    image_index = randi(numexamples);
    x_pos = randi(2048-mask_size);
    sleep_x(i,x_pos:x_pos+mask_size) = X(image_index, x_pos:x_pos+mask_size);
end

end
