% function clusterLabels = alg3(W, D, F, k)
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
% sqrtD = sqrtm(D);
% C = sqrtD\F;
% Ln = (sqrtD\L)/sqrtD;
% Ln = (Ln+Ln')/2;
% sigma = norm(Ln,1);
%     
% try
%     [X, vals] = eigs(@(b) Afun(Ln, C, b, sigma), n, k, 'sr', 'SubspaceDimension',4*k);
% catch
%     [X, vals] = eigs(@(b) Afun(Ln, C, b, sigma), n, k, 'sr','MaxIterations',1000,'SubspaceDimension', 4*k);
% end
% 
% H = sqrtD\X;
% 
% try
%     clusterLabels = kmeans(H,k,'Replicates',10);
% catch
%     clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);
% end
% 
% resF = norm(F'*H, 1)/max(size(F,2),k)
% end

function clusterLabels = alg3(W, D, F, k)
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
sqrtD = sqrtm(D);
C = sqrtD\F;
Ln = (sqrtD\L)/sqrtD;

Ln = (Ln+Ln')/2;
sigma = norm(Ln,1);
% eigvals = eigs(Ln, k, 'sr','MaxIterations',1000,'SubspaceDimension', 4*k)

% s = warning('error', 'MATLAB:EigenvaluesNotConverged');
% warning('error', 'MATLAB:MaxIterationReached');
% try
%     [X, vals] = eigs(@(b) Afun(Ln, C, b, sigma), n, k, 'sr', 'SubspaceDimension',2*k);
% catch
%     fprintf('warning catched\n');
    [X, vals] = eigs(@(b) Afun(Ln, C, b, sigma), n, k, 'sr','MaxIterations',1000,'SubspaceDimension', 4*k);
% end
% warning(s);

H = sqrtD\X;
% scatter(H(:,1),H(:,2))
% plot(H(:,1),H(:,2),'k*','MarkerSize',5);

% try
%     clusterLabels = kmeans(H,k,'Replicates',10);
% catch
%     fprintf('warning catched\n');
    clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);
% end
% Restore the warnings back to their previous (non-error) state
% warning(s);

% tr3 = trace(H'*L*H)
% resH3 = norm(H'*D*H-eye(k),1)/k
% resF3 = norm(F'*H, 1)/max(size(F,2),k)

end