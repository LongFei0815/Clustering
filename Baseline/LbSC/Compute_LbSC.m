function [LSC_clusterlabels,LSC_Avg_Balance,LSC_Avg_Fraction] = Compute_LbSC(data,k,opts)
% label = LSC(data,k,opts): Landmark-based Spectral Clustering

%% 设置默认参数
if (~exist('opts','var'))
   opts = [];
end


p = 30;
if isfield(opts,'p')
    p = opts.p;
end

r = 3;
if isfield(opts,'r')
    r = opts.r;
end

maxIter = 100;
if isfield(opts,'maxIter')
    maxIter = opts.maxIter;
end

numRep = 10;
if isfield(opts,'numRep')
    numRep = opts.numRep;
end

mode = 'kmeans';
if isfield(opts,'mode')
    mode = opts.mode;
end

nSmp=size(data,1);

%% 使用K-means 选择 Landmark,注意kmMaxIter和kmNumRep仅在k-means下可用。
if strcmp(mode,'kmeans')
    kmMaxIter = 5;
    if isfield(opts,'kmMaxIter')
        kmMaxIter = opts.kmMaxIter;
    end
    kmNumRep = 1;
    if isfield(opts,'kmNumRep')
        kmNumRep = opts.kmNumRep;
    end
    [dump,marks]=litekmeans(data,p,'MaxIter',kmMaxIter,'Replicates',kmNumRep);
    clear kmMaxIter kmNumRep
elseif strcmp(mode,'random')
    indSmp = randperm(nSmp);
    marks = data(indSmp(1:p),:);
    clear indSmp
else
    error('mode does not support!');
end

%% 构建Z目的是建造相似性矩阵W
D = EuDist2(data,marks,0);
%% 设置高斯核函数的带宽参数 sigma,用于控制相似度的衰减速度
if isfield(opts,'sigma')
    sigma = opts.sigma;
else
    sigma = mean(mean(D));
end
%% 初始化变量
dump = zeros(nSmp,r);
idx = dump;
%% 找到每个数据点的最近 r 个 Landmark
for i = 1:r
    [dump(:,i),idx(:,i)] = min(D,[],2);
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
    D(temp) = 1e100; 
end
%% 计算高斯核相似度
dump = exp(-dump/(2*sigma^2));
%% 归一化相似度
sumD = sum(dump,2);
%% 构建稀疏矩阵 Z
Gsdx = bsxfun(@rdivide,dump,sumD);
Gidx = repmat([1:nSmp]',1,r);
Gjdx = idx;
Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,p);
%% 图分解，将矩阵Z的每一列归一化
feaSum = full(sqrt(sum(Z,1)));
feaSum = max(feaSum, 1e-12);
Z = Z./feaSum(ones(size(Z,1),1),:);
Z=double(Z);
Z=Z';
fprintf('----------进行相似性矩阵构造Z大小为------------\n');
disp(size(Z));
%% 选取整个数据集的敏感属性Group
Group = readmatrix('H:\SFLSC\databases\Obesity_Sensitive.csv');
%% Z进行奇异值分解SVD
[U,S,V] = mySVD(Z,k+1);
V(:,1) = [];
V=V./repmat(sqrt(sum(V.^2,2)),1,k);
%% 最后对SVD的U进行k-means
LSC_clusterlabels=litekmeans(V,k,'MaxIter',maxIter,'Replicates',numRep);
%% 调用标准对称规范化的谱聚类SC_norm
fprintf('----------LSC的结果------------\n');
%% 调用平衡性 ComputeBalance
LSC_Avg_Balance = LSC_Balance(LSC_clusterlabels, Group);
fprintf('----------LSC_Avg_Balance为------------\n');
disp(LSC_Avg_Balance);
%% 调用得分情况 ComputeFraction
LSC_Avg_Fraction = LSC_Fraction(LSC_clusterlabels, Group);
fprintf('---------- LSC_Avg_Fraction为------------\n');
disp(LSC_Avg_Fraction);



