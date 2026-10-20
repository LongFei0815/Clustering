function [Fair_SC_normalized_clusterlabels ,Fair_SC_normalized_Avg_Balance,Fair_SC_normalized_Avg_Fraction] = Compute_Fair_SC_normalized(data, k)
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
    %% 选取整个数据集的敏感属性Group
    Group = readmatrix('E:\ZhongLongFei_Code\LFSC\databases\ElevatorWX2022_Sensitive.csv');
    Group = Group (1:3000,:);
    %% 调用标准化对称FairSC_normalized
    fprintf('----------Fair_SC_normalized的结果------------\n');
    Fair_SC_normalized_clusterlabels = Fair_SC_normalized(Z, k ,Group);
    %% 调用平衡性 ComputeBalance
    Fair_SC_normalized_Avg_Balance = Fair_SC_Balance(Fair_SC_normalized_clusterlabels ,Group);
    fprintf('----------Fair_SC_normalized_Avg_Balance为------------\n');
    disp(Fair_SC_normalized_Avg_Balance);

    %% 调用得分情况 ComputeFraction
    Fair_SC_normalized_Avg_Fraction= Fair_SC_Fraction(Fair_SC_normalized_clusterlabels , Group);
    fprintf('----------Fair_SC_normalized_Avg_Fraction为------------\n');
    disp(Fair_SC_normalized_Avg_Fraction);
end