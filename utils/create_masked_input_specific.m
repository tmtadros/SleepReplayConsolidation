function [ sleep_x] = create_masked_input_specific(X, numexamples, mask_size)
%Computes the average image and then returns masked versions of that input
sleep_input = reshape(X, numexamples, 28, 28);
sleep_x = zeros(numexamples, 28, 28);
for i = 1:numexamples
    image_index = randi(numexamples);
    x_pos = randi(28-mask_size);
    y_pos = randi(28-mask_size);
    sleep_x(i,x_pos:x_pos+mask_size, y_pos:y_pos + mask_size) = sleep_input(image_index, x_pos:x_pos+mask_size, y_pos:y_pos + mask_size);
end

sleep_x = reshape(sleep_x, numexamples, 784);

end
