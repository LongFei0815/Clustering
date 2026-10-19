function [Z,R2]=RPCMcode(X, feat, opts)
modelstyle=opts.modelstyle;
D=X;
%% Initializing weights of neural networks
weight=initializeweight(X,D,opts);
%%
switch modelstyle    
    case 'RPCM_l1f2'
         [weight, Z, ~] = RPCM_l1f2(X,D,weight,opts);       
end

%% Coding by fast computing the neural networks
H=actfun(weight,feat,opts.act_fun);
R2=H{end}; 