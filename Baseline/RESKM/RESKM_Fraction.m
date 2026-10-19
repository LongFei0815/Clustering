function compute_fractions = RESKM_Fraction(clustering,sensitivity)
%labels = readmatrix('matrix_SC-clusterLabels.csv');
 %gfemale = readmatrix('group_gfemale.csv');
    unilb = unique(clustering);
    k = length(unilb);%根据alg1、alg2、alg3算法中的k值长度来算出每个聚类集群得分情况。
    compute_fractions = zeros(k,1);
    for i = 1:k
        i
        idx = clustering == unilb(i);
        ci = sum(idx)
        vfandci = sum(sensitivity(idx))
        compute_fractions(i) = vfandci/sum(idx);
    end
end

