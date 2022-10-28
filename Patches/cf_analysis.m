function [t1sum,t2sum] = cf_analysis(Wall, X)
% Plots weights with 3 colors: zero pixels, overlapping pixels, and
% non-overlapping pixels
 
no_cond = size(Wall,2);
no_inputs = size(X,1);
no_tasks = no_inputs/2;
overlapping_pixels = find(sum(X,1)==no_inputs);
task = 1;

t1sum = zeros(no_cond,1);
t2sum = zeros(no_cond,1);

for nci = 1:no_cond
    overlapping_pixels = find(sum(X(1:4,:) == 1) == no_inputs); 
    
    Wc = Wall(nci);
    % On pixel locations for each image
    t1im1on_pixels = X(1,:) == 1 & X(2,:) ~= 1;
    t1im2on_pixels = X(2,:) == 1 & X(1,:) ~= 1;
    t2im1on_pixels = X(3,:) == 1 & X(4,:) ~= 1;
    t2im2on_pixels = X(4,:) == 1 & X(3,:) ~= 1;
    
    % weights connecting from on pixels to corresponding output neuron
    t1im1_onweights = Wc{1}(1,t1im1on_pixels);
    t1im2_onweights = Wc{1}(2,t1im2on_pixels);
    t2im1_onweights = Wc{1}(3,t2im1on_pixels);
    t2im2_onweights = Wc{1}(4,t2im2on_pixels);
    
    % weights connecting from on pixels to wrong task neurons
    t1im1_offweights = horzcat(squeeze(Wc{1}(3,t1im1on_pixels)), squeeze(Wc{1}(4,t1im1on_pixels)));
    t1im2_offweights = horzcat(squeeze(Wc{1}(3,t1im2on_pixels)), squeeze(Wc{1}(4,t1im2on_pixels)));
    t2im1_offweights = horzcat(squeeze(Wc{1}(1,t2im1on_pixels)), squeeze(Wc{1}(2,t2im1on_pixels)));
    t2im2_offweights = horzcat(squeeze(Wc{1}(1,t2im2on_pixels)), squeeze(Wc{1}(2,t2im2on_pixels)));
    
    % Concatenate into t1 and t2 weights
    t1Wt1 = horzcat(squeeze(t1im1_onweights), squeeze(t1im2_onweights));
    t1Wt2 = horzcat(squeeze(t1im2_offweights), squeeze(t1im1_offweights));
    t2Wt1 = horzcat(squeeze(t2im1_offweights), squeeze(t2im2_offweights));
    t2Wt2 = horzcat(squeeze(t2im2_onweights), squeeze(t2im1_onweights));
    % Overlapping weights
    overlappingWt1 = horzcat(squeeze(Wc{1}(1,overlapping_pixels)), squeeze(Wc{1}(2, overlapping_pixels)));
    overlappingWt2 = horzcat(squeeze(Wc{1}(3,overlapping_pixels)), squeeze(Wc{1}(4, overlapping_pixels)));

    sumT1 = sum(squeeze(Wc{1}(1,overlapping_pixels))) + sum(squeeze(t1im1_onweights))
    sumT2 = max([sum(squeeze(Wc{1}(3,overlapping_pixels))) sum(squeeze(Wc{1}(4, overlapping_pixels)))]) + max([sum(squeeze(Wc{1}(3,t1im1on_pixels))), sum(squeeze(Wc{1}(4,t1im1on_pixels)))])
    t1sum(nci) = sumT1;
    t2sum(nci) = sumT2;
end