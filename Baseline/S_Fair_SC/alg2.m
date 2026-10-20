% function clusterLabels = alg2(W, D, F, k)
% %INPUT:
% %   W ... (weighted) adjacency matrix of size n x n
% %   D ... degree matrix of W
% %   F ... group membership matrix G of size n x (h-1)
% %   k ... number of clusters
% %
% %OUTPUT:
% % clusterLabels ... vector of length n comprising the cluster label for each
% %                  data point
% n = size(W, 1);
% L = D - W;
% Z = null(F');
% 
% Q=sqrtm(Z'*D*Z);
% 
% M=(Q\Z')*L*(Z/Q); 
% M=(M+M')/2;
% 
% try
%     [Y, vals] = eigs(@(b) Afun2(M, b), n - size(F,2), k,'sr','SubspaceDimension',4*k);
% catch
%     [Y, vals] = eigs(@(b) Afun2(M, b), n - size(F,2), k,'sr','MaxIterations',1000,'SubspaceDimension',4*k);
% end
% 
% H = Z*(Q\Y);
% 
% try
%     clusterLabels = kmeans(H,k,'Replicates',10);
% catch
%     clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);
% end
% 
% end


function clusterLabels = alg2(W, D, F, k)
%INPUT:
%   W ... (weighted) adjacency matrix of size n x n
%   D ... degree matrix of W
%   F ... group membership matrix G of size n x (h-1)
%   k ... number of clusters
%
%OUTPUT:
% clusterLabels ... vector of length n comprising the cluster label for each
%                  data point
n = size(W, 1);
L = D - W;
Z = null(F');

Q=sqrtm(Z'*D*Z);

M=(Q\Z')*L*(Z/Q); 
M=(M+M')/2;

% s = warning('error', 'MATLAB:EigenvaluesNotConverged');
% warning('error', 'MATLAB:MaxIterationReached');
% try
%     [Y, vals] = eigs(@(b) Afun2(M, b), n - size(F,2), k,'sr','SubspaceDimension',2*k);
% catch
%     fprintf('warning catched\n');
    [Y, vals] = eigs(@(b) Afun2(M, b), n - size(F,2), k,'sr','MaxIterations',1000,'SubspaceDimension',4*k);
% end
% warning(s);

H = Z*(Q\Y);

% try
%     clusterLabels = kmeans(H,k,'Replicates',10);
% catch
%     fprintf('warning catched\n');
    clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);
% end
% warning(s);

% tr2 = trace(H'*L*H)
% resH2 = norm(H'*D*H-eye(k),1)/k
% resF2 = norm(F'*H, 1)/max(size(F,2),k)

end

