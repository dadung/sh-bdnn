function mAP = runHash( dataset, basedir, L, dimen)
%RUNHASH run demos
%INPUT
%   dataset: name of dataset, it should be cifar-10 or mnist
%   basedir: folder contains dataset
%   L: number of bits to encoder each vector
%   dimen: size of dimensionality of input features
%OUTPUT
%   mAP: mean average precision
    %% Setup parameters and dataset
    max_iter = 20; % Maximum iteration of alternating optimization B and W
    iter_lbfgs = 400; % number iteration of L-BFGS for learning weights W
    
    lambda1 = 10^-5;
    lambda2 = 5*10^-2;
    lambda3 = 10^-2;
    lambda4 = 0.5*10^-6;
    lambda5 = 10^-2; % please tune between "10^-2" and "0.5*10^-2"
    
    % Configure number of layers 
    switch L
        case 8
            hiddenSize(1) = 90;
            hiddenSize(2) = 20;
        case 16
            hiddenSize(1) = 90;
            hiddenSize(2) = 30;
        case 24
            hiddenSize(1) = 100;
            hiddenSize(2) = 40;
        case 32
            hiddenSize(1) = 120;
            hiddenSize(2) = 50;
        otherwise
            error('please specify L = 8, 16, 24 or 32');
    end
    hiddenSize(3) = L;

    % Load dataset
    [ Xtrain, Xtest, gnd_inds, Ytrain, Ytest] = config_cifar10(basedir, dimen);
    gnd_inds = double(gnd_inds');
    
    % Preprocess data
    numclass = 10; 
    train_sample_per_class = 3000;
    train_sample_per_class_orig = 5000;
    nval = 1000; 
    
    [Xtrain_sub, Ytrain_sub, S, Xval, val_gnd_inds] = construct_train_val_data_cifar( ...
            Xtrain', Ytrain, numclass, train_sample_per_class_orig, ...
            train_sample_per_class, nval);
        
    % Display data info
    fprintf('Dataset = %s\n', dataset);
    fprintf('\t Number of train = %d\n', size(Xtrain_sub, 2));
    fprintf('\t Number of test = %d\n', size(Xtest, 1));
    fprintf('\t Number of validation = %d\n', size(Xval, 1));

    %% Train deep neural networks
    tic;
    stack = learn_all(S, Xtrain, Xtrain_sub, Xval, val_gnd_inds, hiddenSize, ...
        lambda1, lambda2, lambda3, lambda4, lambda5, iter_lbfgs, max_iter);
    fprintf('Train in %.3fs\n', toc);

    %% Evaluation
    Htrain = feedForwardDeep(stack, Xtrain')';
    Htest = feedForwardDeep(stack, Xtest')';
    Btrain = zeros(size(Htrain));
    Btrain(Htrain >= 0) = 1;
    Btest = zeros(size(Htest));
    Btest(Htest >= 0) = 1;

    mAP = KNNMap(Btrain,Btest,size(Btrain,1),gnd_inds) * 100;
    
end

