function [K_means_clusterlabels,K_means_Avg_Balance,K_means_Avg_Fraction] = Compute_K_means(data, k)
    % 输入:
    %   data: 数据集 (nSmp × nFea 矩阵，每行是一个样本)
    %   k: 聚类数量
    % 输出:
    %   clusterLabels: 聚类标签 (nSmp × 1 向量)

    %% 读取敏感属性 Group
    Group = readmatrix('E:\ZhongLongFei_Code\LFSC\databases\ElevatorWX2022_Sensitive.csv');
    Group =Group (1:3000,:);
    %% 调用K_means
    fprintf('---------- K_means的结果------------\n');
    [K_means_clusterlabels,~] = litekmeans(data,k,'Replicates',10, 'MaxIter',500);

    %% 调用平衡性 ComputeBalance
    K_means_Avg_Balance = K_means_Balance(K_means_clusterlabels, Group);
    fprintf('----------  K_means_Avg_Balance为------------\n');
    disp(K_means_Avg_Balance);

    %% 调用得分情况 ComputeFraction
    K_means_Avg_Fraction = K_means_Fraction(K_means_clusterlabels, Group);
    fprintf('---------- K_means_Avg_Fraction为------------\n');
    disp(K_means_Avg_Fraction);
end