function [LSC_normalized_clusterlabels,LSC_normalized_Avg_Balance,LSC_normalized_Avg_Fraction] = Compute_LSC_normalized(data,k,opts)

%% 设置默认参数
if (~exist('opts','var')) %检查 opts 是否存在，如果不存在则初始化为空结构体。
   opts = [];
end


p = 50;             %设置 Landmark 的数量 p,默认值为 200。
if isfield(opts,'p') %如果 opts 中指定了 p,则使用指定的值。
    p = opts.p;
end

r = 5;               %设置每个样本点最近的 Landmark 数量 r,默认值为 6。
if isfield(opts,'r') %如果 opts 中指定了 r,则使用指定的值。
    r = opts.r;
end

maxIter = 500;              %设置最终 K-means 的最大迭代次数 maxIter,默认值为 100。
if isfield(opts,'maxIter')  %如果 opts 中指定了最大迭代次数maxIter ,则使用指定的值。
    maxIter = opts.maxIter;
end

numRep = 10;                %设置最终 K-means 的重复次数 numRep,默认值为 10。
if isfield(opts,'numRep')   %如果 opts 中指定了重复次数numRep ,则使用指定的值。
    numRep = opts.numRep;
end

mode = 'kmeans';            %设置 Landmark 选择方法 mode,默认值为 'kmeans'。
if isfield(opts,'mode')     %如果 opts 中指定了选择方法 mode类型 ,则使用指定的值。
    mode = opts.mode;
end

nSmp=size(data,1);         %获取数据集data的行数，即样本数量给nSmp。


%% 使用K-means 选择 Landmark,注意kmMaxIter和kmNumRep仅在k-means下可用。
if strcmp(mode,'kmeans') %如果 mode 是 'kmeans',则使用 K-means 算法选择 Landmark。
    kmMaxIter = 5;                   %初始kmeans的最大迭代次数（默认为5）
    if isfield(opts,'kmMaxIter')     %如果 opts 中指定了最大迭代次数kmMaxIter ,则使用指定的值。
        kmMaxIter = opts.kmMaxIter;
    end
    kmNumRep = 1;                   %初始kmeans的重复次数（默认为1）
    if isfield(opts,'kmNumRep')     %如果 opts 中指定了重复次数kmNumRep ,则使用指定的值
        kmNumRep = opts.kmNumRep;
    end
    [dump,marks]=litekmeans(data,p,'MaxIter',kmMaxIter,'Replicates',kmNumRep);
    clear kmMaxIter kmNumRep

%% 使用random来选择Landmark，data是nSmp × nFea的数据矩阵，其中每行为一个样本点。
elseif strcmp(mode,'random')    %如果 mode 是 'random',则从数据集中随机选择 p 个样本作为 Landmark。
    indSmp = randperm(nSmp);    %生成一个随机排列的索引向量 indSmp,长度为nSmp=size(data,1)即data的行数，即样本数量。
                                %例如nSmp = 10,则indSmp= randperm(10) 可能生成一个1到10的随机排列索引 [3, 7, 1, 9, 5, 2, 10, 4, 8, 6];
    marks = data(indSmp(1:p),:);%从indSmp中选取前p个索引,再从data中提取这些p个索引对应的样本做为marks∈R(p×nFea),即landmark。
    clear indSmp
else                          %反之 mode 不是 'kmeans' 或 'random',则报错
    error('mode does not support!');
end


%% Z construction构建相似度矩阵 Z
D = EuDist2(data,marks,0);

  %% 设置高斯核函数的带宽参数 sigma,用于控制相似度的衰减速度
if isfield(opts,'sigma')    %如果 opts 中指定了 sigma,则使用指定的值。
    sigma = opts.sigma;
else                        %否则,将 sigma 设置为距离矩阵 D 的平均值
    sigma = mean(mean(D));  %即mean(D)计算D的每一列的平均值，外层再对内层结果计算均值。
end
 %% 初始化变量
dump = zeros(nSmp,r);  %大小为 nSmp × r的全零矩阵用于存储每个样本点到其最近 r 个 Landmark 的距离
idx = dump;            %dump∈R(nSmp × r)赋值给idx
%% 找到每个数据点的最近 r 个 Landmark
for i = 1:r %遍历 r 次,每次找到每个数据点到其最近 Landmark 的距离和索引，并更新距离矩阵 D 以避免重复选择相同的 Landmark
    [dump(:,i),idx(:,i)] = min(D,[],2);
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';%计算最小距离对应的线性索引
    D(temp) = 1e100; %将找到的最小距离对应的位置设置为一个很大的值（1e100）,以便在下次迭代中找到下一个最小距离。
end
%% 计算高斯核相似度
dump = exp(-dump/(2*sigma^2));
%% 归一化相似度
sumD = sum(dump,2);%计算每个样本点到其最近 r 个 Landmark 的相似度之和（按行求和）,得到论文中的度矩阵。
Gsdx = bsxfun(@rdivide,dump,sumD);%将 dump 中的每个相似度除以其对应的sumD中的元素，得到归一化后的相似度。论文中的D的负1/2
%% 构建稀疏矩阵 Z是一个nSmp × p 的稀疏矩阵，表示样本点与 Landmark 之间的相似度
Gidx = repmat([1:nSmp]',1,r);
Gjdx = idx;                  
Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,p);

%% Graph decomposition图分解
 %% 将矩阵Z的每一列归一化
feaSum = full(sqrt(sum(Z,1)));%将Z矩阵每列求和再取平方根,稀疏矩阵转为稠密矩阵
feaSum = max(feaSum, 1e-12);  %对 feaSum 中的每个元素取最大值，确保其不小于 1e-12,避免出现除以零的情况。
Z = Z./feaSum(ones(size(Z,1),1),:);
Z=double(Z);
Z = Z';   %这里Z是500×11741的适用于Fair SC进行SVD
fprintf('----------Z的大小为------------\n');
disp(size(Z));
%% 选取整个数据集的 敏感属性Group
Group = readmatrix('E:\ZhongLongFei_Code\LFSC\databases\ElevatorWX2023_Sensitive.csv');
%% 标准对称规范化的SC
 [RPCM_NFair,Dsqinv] = SC_normalized(Z);
 %clusterLabels = SC_unnorm(Z, k,Group)调用非规范化SC
 disp(size(RPCM_NFair));%确保normNFair行数＜列数，使用第一种确保normNFair'*normNFair进行SVD
% 将归一化后的Z进行奇异值分解(SVD)得到V
[~,~,V ]= mySVD(RPCM_NFair,k+1);%对归一化后的 Z 进行奇异值分解（SVD）,提取前 k+1 个奇异向量
%% Z进行SVD后得到V。去掉V的第一个列（第一个奇异向量),通常对应常数特征，对聚类没有帮助.
V(:,1) = [];
 %% 将V归一化奇异向量
V=V./repmat(sqrt(sum(V.^2,2)),1,k);
%% 将V进行k-means得到聚类标签(标准对称规范化的Fair SC)
H=Dsqinv*V;
LSC_normalized_clusterlabels=litekmeans(H,k,'MaxIter',maxIter,'Replicates',numRep);
 %% 调用平衡性 ComputeBalance
 fprintf('----------LSC_normalized的结果为------------\n');
 LSC_normalized_Avg_Balance = LFSC_Balance(LSC_normalized_clusterlabels, Group);
 fprintf('----------LSC_normalized_Avg_Balance为------------\n');
 disp(LSC_normalized_Avg_Balance);

 %% 调用得分情况 ComputeFraction
 LSC_normalized_Avg_Fraction = LFSC_Fraction(LSC_normalized_clusterlabels, Group);
 fprintf('----------LSC_normalized_Avg_Fraction为------------\n');
 disp(LSC_normalized_Avg_Fraction);







