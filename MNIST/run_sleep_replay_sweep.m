%% MNIST catastrophic forgetting example, GIDO Class task
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));


%%
percent_replay = [0.0 0.01 0.025 0.05 0.1 0.25 0.5 1.0];

num_trials = 5;
accuracy_sleep = zeros(length(percent_replay), num_trials);
accuracy_nosleep = zeros(length(percent_replay), num_trials);
for i = 1:length(percent_replay)
  
    for j = 1:num_trials
        task_order = [1 2 3 4 5];
        accuracy_sleep(i,j) = run_sleep_with_replay(task_order, percent_replay(i), 1);
        accuracy_nosleep(i,j) = run_sleep_with_replay(task_order, percent_replay(i), 0);
    end
end

save('mnist_accuracy_optimized', 'accuracy_sleep', 'accuracy_nosleep')


%% Plot accuracy
% figure;
% num_replayed_examples = [0 100 200 300 400 500 1000 5000];
% error = std(newaccuracy, [], 2);
% hE= errorbar(num_replayed_examples, mean(newaccuracy,2), error); hold on
% set(hE                            , ...
%   'Marker'          , '.'         , ...
%   'Color'           , [.3 .3 .3]  );
% set(hE                            , ...
%   'LineWidth'       , 1           , ...
%   'Marker'          , 'o'         , ...
%   'MarkerSize'      , 6           , ...
%   'MarkerEdgeColor' , [.2 .2 .2]  , ...
%   'MarkerFaceColor' , [.7 .7 .7]  );
% 
% hTitle  = title ('Sleep with Replayed Examples');
% hXLabel = xlabel('Number of replayed examples');
% hYLabel = ylabel('Classification accuracy');
% 
% set(gca                       , ...
%     'FontName'   , 'Helvetica' );
% set([hTitle, hXLabel, hYLabel], ...
%     'FontName'   , 'AvantGarde');
% set([hXLabel, hYLabel]  , ...
%     'FontSize'   , 12          );
% set( hTitle                    , ...
%     'FontSize'   , 14          , ...
%     'FontWeight' , 'bold'      );
% 
% %set(gca, ...
% %  'Box'         , 'off'     , ...
% %  'TickDir'     , 'out'     , ...
% %  'TickLength'  , [.02 .02] , ...
% %  'XMinorTick'  , 'on'      , ...
% %  'YMinorTick'  , 'on'      , ...
% %  'YGrid'       , 'on'      , ...
% %  'XColor'      , [.3 .3 .3], ...
% %  'YColor'      , [.3 .3 .3], ...
% %  'YTick'       , 0:500:2500, ...
% %  'LineWidth'   , 1         );
% 
% 
