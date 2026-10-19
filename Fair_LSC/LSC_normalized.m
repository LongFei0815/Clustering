function  [normalized_N,Dsqinv]= LSC_normalized(adj,~)
%implementation of normalized SC as described in Appendix A 
%
%INPUT:
%adj ... (weighted) adjacency matrix of size n x n
%k ... number of clusters
%
%OUTPUT:
%clusterLabels ... vector of length n comprising the cluster label for each
%                  data point

W = adj'*adj;
n=size(W,1);

degrees = sum(W, 1);
D = diag(degrees);
Dsqinv=diag(1./sqrt(degrees));

normalized_N = adj*Dsqinv;
M=normalized_N'*normalized_N;
M=(M+M')/2;


%try
   % [H, eigValues] = eigs(M,k,'smallestabs','MaxIterations',500,'SubspaceDimension',min(n,max(2*k,25)));
%catch
    %[H, eigValues] = eigs(M,k,'smallestreal','MaxIterations',1000,'SubspaceDimension',min(n,max(2*k,25)));
%end
%maxIter = 100;
%numRep = 10; 
%clusterLabels=litekmeans(Dsqinv*H,k,'MaxIter',maxIter,'Replicates',numRep);
end




