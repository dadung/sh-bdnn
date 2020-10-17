%% deepAE_Hash Exercise
function [stackout,deepAEOptTheta,netconfig, cost] = learnDeepNN(trainData,B, S,inputSize,hiddenSize,...
    lambda1, lambda2, lambda3, lambda4, lambda5, pre_training, stack, t, iter_lbfgs)
if pre_training
    %% Initialize W1
    sae1OptTheta = initializeParameters(hiddenSize(1), inputSize, trainData);

    
    %% Initialize W2
    [sae1Features] = feedForwardDeepNN(sae1OptTheta, hiddenSize(1), inputSize, trainData);
    sae2OptTheta = initializeParameters(hiddenSize(2), hiddenSize(1), sae1Features);
    
    %% Initialize W3
    [sae2Features] = feedForwardDeepNN(sae2OptTheta, hiddenSize(2), hiddenSize(1), sae1Features);
    sae3OptTheta = initializeParameters(hiddenSize(3), hiddenSize(2), sae2Features);
   
    % Stack all initialized (W,c)
    stack{1}.w = reshape(sae1OptTheta(1:hiddenSize(1)*inputSize), hiddenSize(1), inputSize);
    stack{1}.b = sae1OptTheta(2*hiddenSize(1)*inputSize+1:2*hiddenSize(1)*inputSize+hiddenSize(1));
    stack{2}.w = reshape(sae2OptTheta(1:hiddenSize(2)*hiddenSize(1)), hiddenSize(2), hiddenSize(1));
    stack{2}.b = sae2OptTheta(2*hiddenSize(2)*hiddenSize(1)+1:2*hiddenSize(2)*hiddenSize(1)+hiddenSize(2));
    stack{3}.w = reshape(sae3OptTheta(1:hiddenSize(3)*hiddenSize(2)), hiddenSize(3), hiddenSize(2));
    stack{3}.b = sae3OptTheta(2*hiddenSize(3)*hiddenSize(2)+1:2*hiddenSize(3)*hiddenSize(2)+hiddenSize(3));
end
%%======================================================================
%% Train Deep NN by back propagation
[stackparams, netconfig] = stack2params(stack); % stack all parametter to a vector
deepNNTheta = stackparams;
options.Method = 'lbfgs';
options.maxIter = iter_lbfgs;	  % 400 Maximum number of iterations of L-BFGS to run 
options.display = 'off';

% run L-BFGS
[deepAEOptTheta, cost] = minFunc( @(p) deepNNCost(p, inputSize, hiddenSize, netconfig, ...
                                   lambda1, lambda2, lambda3, lambda4, lambda5, B, S, trainData), ...
                              deepNNTheta, options);

stackout = params2stack(deepAEOptTheta(1:end), netconfig);
