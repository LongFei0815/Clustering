function [SC_unnormalized_clusterlabels,SC_unnormalized_Avg_Balance,SC_unnormalized_Avg_Fraction] = Compute_SC_unnormalized(data, k)
    % 输入:
    %   data: 数据集 (nSmp × nFea 矩阵，每行是一个样本)
    %   k: 聚类数量
    % 输出:
    %   clusterLabels: 聚类标签 (nSmp × 1 向量)


    %% 构建相似度矩阵 Z
    % 计算欧式距离矩阵 D
    D = EuDist2(data, data, 0); % 不取平方根，减少计算时间

    % 设置高斯核函数的带宽参数 sigma
    sigma = mean(mean(D)); % 使用距离矩阵的均值作为带宽参数

    % 计算高斯核相似度
    Z = exp(-D / (2 * sigma^2)); % 高斯核相似度矩阵

    % 归一化相似度矩阵
    sumD = sum(Z, 2); % 按行求和，得到度矩阵的对角元素
    Z = bsxfun(@rdivide, Z, sumD); % 归一化

    %% 图分解
    % 将矩阵 Z 的每一列归一化
    feaSum = full(sqrt(sum(Z, 1))); % 每列求和并取平方根
    feaSum = max(feaSum, 1e-12); % 避免除以零
    Z = Z ./ feaSum(ones(size(Z, 1), 1), :); % 归一化
    fprintf('----------进行相似性矩阵构造Z大小为------------\n');
    disp(size(Z));
    %% 读取敏感属性 Group
        Group = readmatrix('H:\SFLSC\databases\Hcv_data_Sensitive.csv');
    %% 调用非标准化SC_unnormalized
    fprintf('----------SC_unnorm的结果------------\n');
    SC_unnormalized_clusterlabels = SC_unnormalized(Z, k);%调用非规范化SC

    %% 调用平衡性 ComputeBalance
    SC_unnormalized_Avg_Balance = SC_Balance(SC_unnormalized_clusterlabels, Group);
    fprintf('---------- SC_unnorm_Avg_Balance为------------\n');
    disp(SC_unnormalized_Avg_Balance);

    %% 调用得分情况 ComputeFraction
    SC_unnormalized_Avg_Fraction = SC_Fraction(SC_unnormalized_clusterlabels, Group);
    fprintf('----------SC_unnormalized_Avg_Fraction为------------\n');
    disp(SC_unnormalized_Avg_Fraction);
end