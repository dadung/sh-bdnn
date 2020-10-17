function [ Xtrain, Xtest, gnd_inds, Ytrain, Ytest ] = config_cifar10( base_dir, dimen )
%CONFIG_CIFAR10_2 
    load([base_dir '/cifar_alexnet_' num2str(dimen) '.mat']);
    Ytrain = double(Ytrain); Xtrain = double(Xtrain);
    Ytest = double(Ytest); Xtest = double(Xtest);
    ntrain = size(Xtrain,1);
    nquery = size(Xtest,1);
     %% construct groundtruth 
    ngnd = ntrain/10;
    gnd_inds = zeros(ngnd, nquery);
    for label = 1:10
        gnd = find(Ytrain == label);
        idx = find(Ytest == label);
        gnd_inds(:, idx) = gnd*ones(1,length(idx));
    end
   
   
    
end

