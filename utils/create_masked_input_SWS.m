function [ sleep_x] = create_masked_input_SWS(X, numexamples, mask_size, numON, numOFF)
%Computes the average image and then returns masked versions of that input
sleep_input = mean(X).'; 
sleep_input = reshape(sleep_input, 28, 28);
sleep_x = zeros(28, 28, numexamples*(numON+numOFF));

ind = 1;
for i = 1:numexamples
    x_pos = randi(28-mask_size);
    y_pos = randi(28-mask_size);
    sleep_x(x_pos:x_pos+mask_size, y_pos:y_pos + mask_size, ind:ind+numON-1) = repmat(sleep_input(x_pos:x_pos+mask_size, y_pos:y_pos + mask_size), 1,1,numON);
    ind = ind + numON + numOFF;
end

sleep_x = reshape(sleep_x, 784, numexamples*(numON+numOFF));


end

