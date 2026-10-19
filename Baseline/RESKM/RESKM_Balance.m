function compute_balance = RESKM_Balance(clustering,sensitivity)
   % 输入:
    %   sensitivity: 敏感属性向量 (n x 1 或 1 x n)
    %   clustering: 聚类标签向量 (n x 1 或 1 x n)
    % 输出:
    %   res: 所有簇的平均平衡性

    % 确保 sensitivity 和 clustering 是行向量
    n = length(sensitivity);
    if sum(size(sensitivity) == [n, 1]) == 2
        sensitivity = reshape(sensitivity, [1, n]);
    end
    if sum(size(clustering) == [n, 1]) == 2
        clustering = reshape(clustering, [1, n]);
    end

    % 获取敏感属性和聚类标签的唯一值
    unique_sensitivity = unique(sensitivity);
    unique_clustering = unique(clustering);
    num_S = length(unique_sensitivity); % 敏感属性的类别数
    num_C = length(unique_clustering); % 簇的数量

    % 初始化平衡性
    balance = 0;

    % 遍历每个簇
    for i = 1:num_C
        % 找到当前簇的样本索引
        % 找到当前簇的样本索引
        idx_Ci = (clustering == unique_clustering(i));
        attr_in_Ci = sensitivity(idx_Ci); % 当前簇的敏感属性

        % 初始化当前簇的平衡性
        balance_i = Inf;

        % 遍历敏感属性的所有类别对
        for j = 1:num_S
            for k = 1:num_S
                if k ~= j
                    % 计算类别 j 和类别 k 的数量比例
                    num_attr_j = sum(attr_in_Ci == unique_sensitivity(j));
                    num_attr_k = sum(attr_in_Ci == unique_sensitivity(k));
                    tmp = num_attr_j / num_attr_k;

                    % 更新当前簇的平衡性
                    if tmp < balance_i
                        balance_i = tmp;
                    end
                end
            end
        end

        % 累加当前簇的平衡性
        balance = balance + balance_i;
    end

    % 计算平均平衡性
    balance_avg = balance / num_C;
    compute_balance = balance_avg;
end