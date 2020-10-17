function [traindata, testdata, Ytrain, Ytest, cateTrainTest] = prepare_dataset(basedir, dimen)
% dataset is stored in a row-wise matrix
    load([basedir '/cifar_alexnet_' num2str(dimen) '.mat']);
    Ytrain = double(Ytrain); Xtrain = double(Xtrain);
    Ytest = double(Ytest); Xtest = double(Xtest);
   
    % Normalize all feature vectors to unit length
    %traindata = normalize(double(Xtrain));
    %testdata  = normalize(double(Xtest));
    traindata = double(Xtrain);
    testdata = double(Xtest);

    cateTrainTest = bsxfun(@eq, Ytrain, Ytest'); % traingnd and testgnd are the labels.

end




