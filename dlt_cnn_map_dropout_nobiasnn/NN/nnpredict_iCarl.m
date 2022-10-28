function labels = nnpredict_iCarl(nn, x, S, Sy)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    activations = nn.a{length(nn.a)-1};
    
    nn2 = nnff(nn, S, zeros(size(S,1), nn.size(end)));
    mean_activations = nn2.a{length(nn2.a)-1};
    [~,Slabels] = max(Sy, [], 2);
    
    num_classes = size(Sy,2);
    mean_class_activations = zeros(num_classes,size(activations,2));

    predictions = zeros(length(x), size(Sy,2));
    for i = 1:num_classes
        mean_class_activations(i,:) = mean(mean_activations(Slabels==i,:),1);
        predictions(:,i) = sqrt(sum((activations-mean_class_activations(i,:)).^2,2));
    end
    
    [dummy,i] = max(activations * mean_class_activations', [], 2);
    labels = i;
    
    [C,indices] = min(predictions,[],2);
    labels2 = indices;
    labels = labels2;

end
