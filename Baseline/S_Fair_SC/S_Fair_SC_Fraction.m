%对应Table1中的得分情况:
% gmale是男性保护组|v2并上C2|/|C2|得分
%                 |v2并上C1|/|C1|得分
% gfemale是女性保护组|v1并上C2|/|C2|得分
%                   |v1并上C1|/|C1|得分

function compute_fractions = S_Fair_SC_Fraction(clustering,sensitivity)
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

