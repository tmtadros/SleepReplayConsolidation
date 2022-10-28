function [ sleep_x] = create_masked_input_SWS_CUB(X, numexamples, mask_size, numON, numOFF)
%Computes the average image and then returns masked versions of that input
sleep_input = mean(X).'; 
num_features = length(sleep_input);
sleep_x = zeros(num_features, numexamples*(numON+numOFF));

ind = 1;
for i = 1:numexamples
    x_pos = randi(num_features-mask_size);
    sleep_x(x_pos:x_pos+mask_size,i:i+numON-1) = repmat(sleep_input(x_pos:x_pos+mask_size), 1,numON);
    ind = ind + numON + numOFF;
end

sleep_x = reshape(sleep_x, num_features, numexamples*(numON+numOFF));


end