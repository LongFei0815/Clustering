function  SC_normalized_clusterlabels= SC_normalized(W,k)
%implementation of normalized SC as described in Appendix A 
%
%INPUT:
%adj ... (weighted) adjacency matrix of size n x n
%k ... number of clusters
%
%OUTPUT:
%clusterLabels ... vector of length n comprising the cluster label for each
%                  data point

n = size(W, 1);
degrees = sum(W, 1);
D = diag(degrees);
L = D - W;
sqrtD = sqrtm(D);
Ln = (sqrtD\L)/sqrtD;
Ln = (Ln+Ln')/2;

try
    [Y, eigValues] = eigs(Ln,k,'smallestabs','MaxIterations',500,'SubspaceDimension',min(size(Ln,1),max(2*k,25)));
catch
    [Y, eigValues] = eigs(Ln,k,'smallestreal','MaxIterations',1000,'SubspaceDimension',min(size(Ln,1),max(2*k,25)));
end
   
   H = sqrtD\Y;

SC_normalized_clusterlabels = litekmeans(H,k,'Replicates',10, 'MaxIter',500);
end




