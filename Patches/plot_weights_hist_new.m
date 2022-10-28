function [weights] = plot_weights_hist_new(Wall, X)
% Plots weights with 3 colors: zero pixels, overlapping pixels, and
% non-overlapping pixels
 
no_cond = size(Wall,2);
no_inputs = size(X,1);
no_tasks = no_inputs/2;
overlapping_pixels = find(sum(X,1)==no_inputs);
task = 1;
figure;
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

    subplot(no_cond, no_tasks, task);
    histogram(t1Wt1,5,'FaceColor',[0.4660 0.6740 0.1880]	,'FaceAlpha', 0.5); hold on;
    histogram(overlappingWt1,5,'FaceColor', [0.4940 0.1840 0.5560],'FaceAlpha', 0.5); hold on;
    histogram(t2Wt1,5,'FaceColor', [0.9100    0.4100    0.1700],'FaceAlpha', 0.5); hold on;
    ylim([0 15]);
    xlim([-0.15 0.25]);

    if task == 1 && nci == 1
        title('T1', 'FontSize', 20);
        ylabel('Train T1', 'FontSize', 20);
        legend({'T1-pixels', 'Ovrlp-pixels', 'T2-pixels'}, 'FontSize', 10);
    end
    if task == 3 && nci == 2
        ylabel('Train T2', 'FontSize', 20);
    end
    if task == 5 && nci == 3
        ylabel('After SRA', 'FontSize', 20);
        xlabel('Weight Distribution', 'FontSize', 20)
    end
    subplot(no_cond, no_tasks, task + 1);
    histogram(t1Wt2,5,'FaceColor',[0.4660 0.6740 0.1880]	,'FaceAlpha', 0.5); hold on;
    histogram(overlappingWt2,'FaceColor', [0.4940 0.1840 0.5560],'FaceAlpha', 0.5); hold on;
    histogram(t2Wt2,5,'FaceColor', [0.9100    0.4100    0.1700],'FaceAlpha', 0.5); hold on;
    ylim([0 15]);
        xlim([-0.15 0.25]);
    if task == 1 && nci == 1
        title('T2', 'FontSize', 20);
    end
    if task == 5 && nci == 3
        xlabel('Weight Distribution', 'FontSize', 20)
        xlim([-0.4 0.25]);
    end
    task = task + 2;

end
figure;
histogram(overlappingWt2,'FaceColor', [0.4940 0.1840 0.5560],'FaceAlpha', 0.5); hold on;
histogram(t2Wt2,5,'FaceColor', [0.9100    0.4100    0.1700],'FaceAlpha', 0.5); hold on;
