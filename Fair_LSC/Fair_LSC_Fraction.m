function compute_fractions = Fair_LSC_Fraction(clustering,sensitivity)
    unilb = unique(clustering);
    k = length(unilb);
    compute_fractions = zeros(k,1);
    for i = 1:k
        i
        idx = clustering == unilb(i);
        ci = sum(idx)
        vfandci = sum(sensitivity(idx))
        compute_fractions(i) = vfandci/sum(idx);
    end
end

