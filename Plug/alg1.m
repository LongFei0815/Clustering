% function clusterLabels = alg1(W, D, k)
% %implementation of normalized SC
% %
% %INPUT:
% %   W ... (weighted) adjacency matrix of size n x n
% %   D ... degree matrix of W
% %   k ... number of clusters
% %
% %OUTPUT:
% %   clusterLabels ... vector of length n comprising the cluster label for 
% %                  each data point
%  
% n = size(W, 1);
% L = D - W;
% % componet-wise: 
% % observed slower than sqrtm (is sqrtm smart enough to detect diagonal matrix?)
% sqrtD = sqrtm(D);
% Ln = (sqrtD\L)/sqrtD;
% Ln = (Ln+Ln')/2;
% 
% try
%     [Y, vals] = eigs(@(b) Afun2(Ln, b), n, k, 'sr','SubspaceDimension',4*k);
% catch
%     [Y, vals] = eigs(@(b) Afun2(Ln, b), n, k, 'MaxIterations',1000, 'sr','SubspaceDimension',4*k);
% end
% 
% H = sqrtD\Y;
% 
% try
%     clusterLabels = kmeans(H,k,'Replicates',10);
% catch
%     clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);
% end
% 
% resH1 = norm(H'*D*H-eye(k),1)/k;
% end

function clusterLabels = alg1(W, D, k)

%implementation of normalized SC
%
%INPUT:
%   W ... (weighted) adjacency matrix of size n x n
%   D ... degree matrix of W
%   k ... number of clusters
%
%OUTPUT:
%   clusterLabels ... vector of length n comprising the cluster label for 
%                  each data point
 
n = size(W, 1);
L = D - W;
% componet-wise: 
% observed slower than sqrtm (is sqrtm smart enough to detect diagonal matrix?)
sqrtD = sqrtm(D);
Ln = (sqrtD\L)/sqrtD;
Ln = (Ln+Ln')/2;

% s = warning('error', 'MATLAB:EigenvaluesNotConverged');
% warning('error', 'MATLAB:MaxIterationReached');
% try
%     [Y, vals] = eigs(@(b) Afun2(Ln, b), n, k, 'sr','SubspaceDimension',2*k);
% catch
%     fprintf('warning catched\n');
    [Y, vals] = eigs(@(b) Afun2(Ln, b), n, k, 'sr', 'MaxIterations',1000, 'SubspaceDimension',4*k);
% end
% warning(s);

H = sqrtD\Y;

% try
%     clusterLabels = kmeans(H,k,'Replicates',10);
% catch
%     fprintf('warning catched\n');
    clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);
% end
% warning(s);

% tr1 = trace(H'*L*H)
% resH1 = norm(H'*D*H-eye(k),1)/k
end


