function  LSC_unnormalized_clusterlabels= LSC_unnormalized(adj,~)
%implementation of normalized SC as described in Appendix A 
%
%INPUT:
%adj ... (weighted) adjacency matrix of size n x n
%k ... number of clusters
%
%OUTPUT:
%clusterLabels ... vector of length n comprising the cluster label for each
%                  data point

W = adj*adj;
n = size(W, 1);
degrees = sum(W, 1);
D = diag(degrees);
L = D - W;
L = (L+L')/2; 



%try
    %[Y, eigValues] = eigs(L,k,'smallestabs','MaxIterations',500,'SubspaceDimension',min(size(L,1),max(2*k,25)));
%catch
    %[Y, eigValues] = eigs(L,k,'smallestreal','MaxIterations',1000,'SubspaceDimension',min(size(L,1),max(2*k,25)));
%end
   
   %H = Y;

 %LSC_unnormalized_clusterlabels = litekmeans(H,k,'Replicates',10, 'MaxIter',500);
end




