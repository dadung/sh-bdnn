function MAP = runSDH(dataset, basedir, L, dimen)



[traindata, testdata, traingnd, testgnd, cateTrainTest] = prepare_dataset(basedir, dimen);

traindata = double(traindata);
testdata = double(testdata);


if sum(traingnd == 0)
    traingnd = traingnd + 1;
    testgnd = testgnd + 1;
end

Ntrain = size(traindata,1);
% Use all the training data
X = traindata;
label = double(traingnd);

% get anchors
n_anchors = 1000;
% rand('seed',1);
anchor = X(randsample(Ntrain, n_anchors),:);

% % determin rbf width sigma
% Dis = EuDist2(X,anchor,0);
% % sigma = mean(mean(Dis)).^0.5;
% sigma = mean(min(Dis,[],2).^0.5);
% clear Dis
%sigma = 0.4; % for normalized data
Dis = sqdist(X,anchor);
sigma = mean(min(Dis,[],2).^0.5);
PhiX = exp(-sqdist(X,anchor)/(2*sigma*sigma));
PhiX = [PhiX, ones(Ntrain,1)];

Phi_testdata = exp(-sqdist(testdata,anchor)/(2*sigma*sigma)); clear testdata
Phi_testdata = [Phi_testdata, ones(size(Phi_testdata,1),1)];
Phi_traindata = exp(-sqdist(traindata,anchor)/(2*sigma*sigma)); clear traindata;
Phi_traindata = [Phi_traindata, ones(size(Phi_traindata,1),1)];

% learn G and F
maxItr = 5;
gmap.lambda = 1; gmap.loss = 'L2';
Fmap.type = 'RBF';
Fmap.nu = 1e-5; %  penalty parm for F term
Fmap.lambda = 1e-2;

%% run algo
nbits = L;

% Init Z
randn('seed',3);
Zinit=sign(randn(Ntrain,nbits));

debug = 0;
[~, F, H] = SDH(PhiX,label,Zinit,gmap,Fmap,[],maxItr,debug);

%% evaluation

AsymDist = 0; % Use asymmetric hashing or not

if AsymDist 
    H = H > 0; % directly use the learned bits for training data
else
    H = Phi_traindata*F.W > 0;
end

tH = Phi_testdata*F.W > 0;

B = compactbit(H);
tB = compactbit(tH);

hammTrainTest = hammingDist(tB, B)';

% hamming ranking: MAP
[~, HammingRank]=sort(hammTrainTest,1);
MAP = cat_apcal(traingnd,testgnd,HammingRank)*100;















