function [Z,N] = Fair_LSC_unnormalized(adj,sensitive)
%implementation of fair unnormalized SC as stated in Alg. 2 
%
%INPUT:
%adj ... (weighted) adjacency matrix of size n x n
%k ... number of clusters
%sensitive ... vector of length n encoding the sensitive attribute 
%
%OUTPUT:
%clusterLabels ... vector of length n comprising the cluster label for each
%                  data point

W = adj'*adj;
n = size(adj, 1);

% converting sensitive to a vector with entries in [h] and building F %%%
sens_unique=unique(sensitive);
h = length(sens_unique);
sens_unique=reshape(sens_unique,[1,h]);

sensitiveNEW=sensitive;

temp=1;
for ell=sens_unique
    sensitiveNEW(sensitive==ell)=temp;
    temp=temp+1;
end
    
F=zeros(n,h-1);

for ell=1:(h-1)
    temp=(sensitiveNEW == ell);
    F(temp,ell)=1; 
    groupSize = sum(temp);
    F(:,ell) = F(:,ell)-groupSize/n;
end
%%%%

%degrees = sum(W, 1);
%D = diag(degrees);
%L = D-W;

Z = null(F');
N=adj*Z;
M=N'*N;
M=(M+M')/2;

%try
   % [Y, eigValues] = eigs(Msymm,k,'smallestabs','MaxIterations',500,'SubspaceDimension',min(size(Msymm,1),max(2*k,25)));
%catch
    %[Y, eigValues] = eigs(Msymm,k,'smallestreal','MaxIterations',1000,'SubspaceDimension',min(size(Msymm,1),max(2*k,25)));
%end

%H = Z*Y;

%clusterLabels = kmeans(H,k,'Replicates',10);
end

