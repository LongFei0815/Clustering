function [train_d, train_t] = selectsrno(featureMat, labelMat, selectnum, style)
% 安全版本样本选择函数
% 输入:
%   featureMat - d×n 特征矩阵
%   labelMat   - 1×n 标签向量
%   selectnum  - 每类选择样本数
%   style      - 0:全局随机, 1:按类随机
% 输出:
%   train_d    - 选择的特征
%   train_t    - 对应的标签

    % 输入验证
    validateattributes(featureMat, {'numeric'}, {'2d'});
    validateattributes(labelMat, {'numeric'}, {'size', [1, size(featureMat,2)]});
    validateattributes(selectnum, {'numeric'}, {'positive', 'integer'});
    
    if style == 0
        % 全局随机选择
        Y = randperm(size(labelMat, 2));
        Ytrain = sort(Y(1:min(selectnum, end)));
        train_d = featureMat(:, Ytrain);
        train_t = labelMat(:, Ytrain);
        
    elseif style == 1
        % 按类别随机选择
        unique_labels = unique(labelMat);
        labelnum = length(unique_labels);
        
        % 预分配内存（提升性能）
        max_possible_samples = labelnum * selectnum;
        train_d = zeros(size(featureMat,1), max_possible_samples);
        train_t = zeros(1, max_possible_samples);
        current_pos = 1;
        
        for i = 1:labelnum
            % 获取当前类别样本索引
            class_idx = (labelMat == unique_labels(i));
            class_features = featureMat(:, class_idx);
            class_labels = labelMat(:, class_idx);
            
            % 安全选择
            n_in_class = size(class_features, 2);
            actual_select = min(selectnum, n_in_class);
            
            if actual_select < selectnum
                warning('类别%d仅选择%d/%d个样本', ...
                        unique_labels(i), actual_select, selectnum);
            end
            
            % 随机选择
            kk = randperm(n_in_class, actual_select);
            selected_range = current_pos : current_pos + actual_select - 1;
            
            train_d(:, selected_range) = class_features(:, kk);
            train_t(:, selected_range) = class_labels(:, kk);
            
            current_pos = current_pos + actual_select;
        end
        
        % 裁剪多余预分配空间
        train_d = train_d(:, 1:current_pos-1);
        train_t = train_t(:, 1:current_pos-1);
    else
        error('无效style值: 只支持0或1');
    end
end