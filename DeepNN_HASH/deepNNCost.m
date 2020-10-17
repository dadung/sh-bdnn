function [ cost, grad, J2, J3, J4, J5, J6 ] = deepNNCost(theta, inputSize, hiddenSize, netconfig, ...
    lambda1, lambda2, lambda3, lambda4, lambda5, B, S, data)
                                         
%% Extract stack 
% Extract out the "stack"
stack = params2stack(theta(1:end), netconfig);

stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this
m = size(data, 2); %number of data

%% Perform forward pass to compute activation a2, a3, a4, a5 for layer L2, L3, L4, L5  

a{1} = data;
X = data;
z{1} = [];
nl = numel(stack) + 1; %number of layers
for d = 1:numel(stack)
    z{d+1} = bsxfun(@plus, stack{d}.w * a{d}, stack{d}.b);
    if d+1 <= 3 %sigmoid activation function
        a{d+1} = vectorized_f(z{d+1},'sigmoid');
    else 
        if d+1 == 4
%             a{d+1} = vectorized_f(z{d+1},'tanh'); %tanh
            a{d+1} = z{d+1};
        else %linear (identity activation function)
            a{d+1} = z{d+1};
        end
    end
end

%% compute J (cost function)

J2 = 0;
for l = 1:nl-1
    J2 = J2 + mnorm(stack{l}.w);
end
J2 = (lambda1/2)*J2;
J3 = ( lambda2/(2*m) )* mnorm(a{nl}-B);
tmp = a{nl}*a{nl}';
tmp_2 = (1/hiddenSize(3))* ( a{nl}'*a{nl} );
J4 = (lambda3/2) * mnorm( (1/m)*tmp - eye(size(tmp,1)) );
J5 = (lambda4/(2*m)) * mnorm (a{nl}*ones(m,1));
J6 = (lambda5/(2*m)) * mnorm(tmp_2 - S);

cost = J2 + J3 + J4 + J5 + J6;

%% compute gradient of J w.r.t. others w
 delta{nl} = (lambda2/m) * (a{nl} - B) + (2*lambda3/m)*( (1/m)*tmp - eye(size(tmp,1)) ) * a{nl} ...
     + (lambda4/m)*(a{nl}*ones(m,1)*ones(1,m)) + ...
     (lambda5/m) * (1/hiddenSize(3)) * a{nl} * (( tmp_2 - S) + (tmp_2 - S)'); %if at layer (nl-1) using identity activation f(z{nl-1}) = z{nl-1} --> f'(z{nl-1}) = 1


for l = (nl-1):-1:2
    delta{l} = (stack{l}.w' * delta{l+1}) .* fprime(z{l},'sigmoid');
end

for l = (nl-1):-1:1
    stackgrad{l}.w = delta{l+1} * a{l}' + lambda1 * stack{l}.w;
    stackgrad{l}.b = sum(delta{l+1},2); % = delta{l+1} * ones(m,1)
end
% -------------------------------------------------------------------------

%% Roll gradient vector
[grad, netconfig] = stack2params(stackgrad);
end
