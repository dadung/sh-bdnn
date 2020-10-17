function [ Xtrain_sub, Ytrain_sub, S, Xval, val_gnd_inds] = construct_train_val_data_cifar( Xtrain, Ytrain, ...
    numclass, sample_per_class_orig, sample_per_class, nval)
%construct_train_val_data_cifar randomly pick training data and
% randomly pick the validation data from original training data

    nval_per_class = nval / numclass;
    Xtrain_sub = zeros(size(Xtrain,1),numclass*sample_per_class);
    Ytrain_sub = zeros(numclass*sample_per_class,1);
    S = -ones(numclass*sample_per_class); % init similarity matrix
    Xval = zeros(size(Xtrain,1), nval); % init validation set
    Yval = zeros(nval, 1);
    val_gnd_inds = zeros(nval, sample_per_class_orig);
    count1 = 1;
    count_val = 1;
    for i = 1:numclass
        % pick data from a specific class
        idx = find(Ytrain == i);
        X = Xtrain(:,idx);
        
        % random the indices
        idxsub = randperm(size(X,2));
        
        % get training data
        Xtrain_sub(:,count1:count1+sample_per_class-1) = X(:,idxsub(1:sample_per_class));
        Ytrain_sub(count1:count1+sample_per_class-1) = i;
        
        % construct similarity matrix
        S(count1:count1+sample_per_class-1, count1:count1+sample_per_class-1) = 1;
        
        % get validation data
        Xval(:, count_val:count_val+nval_per_class-1) = ...
            X(:,idxsub(sample_per_class+1:sample_per_class+nval_per_class));
        Yval(count_val:count_val+nval_per_class-1) = i;
        
        % construct groundtruth for validation data
        gnd = find(Ytrain == i);
        gnd = gnd*ones(1, nval_per_class); % duplicate for assignment
        val_gnd_inds(count_val:count_val+nval_per_class-1,:) = gnd';
        
        count_val = count_val + nval_per_class;
        count1 = count1 + sample_per_class;
        
        
    end
    Xval = Xval';
end

