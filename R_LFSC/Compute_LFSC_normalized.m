function [LFSC_normalized_clusterlabels,LFSC_normalized_Avg_Balance,LFSC_normalized_Avg_Fraction] = Compute_LFSC_normalized(data,k,opts)
%这段代码是计算 data 进行聚类,返回聚类结果 label。
%res = FairSC_normalized_LSC(Z, nCluster, opt)这里accnuctLSC.m函数调用了LSC函数
%将参数Z传递给data、nCluster聚类数量传递给k、opts保持不变，最后将聚类结果label传递给res

%%Input:
%%      data: 大小为nSmp × nFea的数据矩阵，其中每行为一个样本点
%%         k: 簇的数量
%%      opts: 算法的选项,其中
%%                 p: 选择的landmark数量（默认为500）   
%%                 r: 表示距离最近的landmark数目（默认为3）
%%            numRep: 最终kmeans的重复次数（默认为10）
%%           maxIter: 最终kmeans的最大迭代次数（默认为100）
%%              mode: landmark选择方法，目前支持k-means和random
%%           k-means: 使用kmeans生成的集群中心（默认）
%%            random: 从原始数据集中随机抽取数据点
%% 以下参数仅在"kmeans"模式下有效:
%%          kmNumRep: 初始kmeans的重复次数（默认为）
%%         kmMaxIter: 初始kmeans的最大迭代次数（默认为5）
%%Output:
%%             label: 每个数据点的聚类分配
%%需要调用litekmeans.m函数
%%用法
%%             data = rand([100,50]);
%%            label = LSC(data,10);
%%可以参考文章:Large Scale Spectral Clustering with Landmark-Based Representation


%% 设置默认参数
if (~exist('opts','var')) %检查 opts 是否存在，如果不存在则初始化为空结构体。
   opts = [];
end


p = 30;             %设置 Landmark 的数量 p,默认值为 200。
if isfield(opts,'p') %如果 opts 中指定了 p,则使用指定的值。
    p = opts.p;
end

r = 3;               %设置每个样本点最近的 Landmark 数量 r,默认值为 6。
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
%data大小为nSmp × nFea的数据矩阵，其中每行为一个样本点

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
    %调用litekmeans函数对数据data进行k-means聚类。
    %参数:data为输入的数据矩阵nSmp × nFea、p聚类簇数(即landmark的数量)。
    %'MaxIter'是K-means的最大迭代次数、kmMaxIter是初始kmeans的最大迭代次数
    %'Replicates'是设置 K-means 的重复次数,kmNumRep是初始kmeans的重复次数
    %最后返回值:dump是每个样本点到其最近聚类中心的距离、marks是聚类中心（即 Landmark）
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

%% 构建Z目的是建造相似性矩阵W
%%% 计算数据集data和marks(landmark)之间的欧氏距离矩阵D(不对D取平方根,减少计算时间),所以设置了0传递给EuDist2函数中的bSqrt值。
%%% 标准的欧式距离是要对D取平方根的，即设置为1传递给EuDist2函数中的bSqrt值。
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
sumD = sum(dump,2);
Gsdx = bsxfun(@rdivide,dump,sumD);%将 dump 中的每个相似度除以其对应的sumD中的元素，得到归一化后的相似度。论文中的D的负1/2
%% 构建稀疏矩阵 Z是一个nSmp × p 的稀疏矩阵，表示样本点与 Landmark 之间的相似度
Gidx = repmat([1:nSmp]',1,r);%生成行索引矩阵，大小为 nSmp × r(r是复制的次数)，每行是样本点的索引，每列是[1:nSmp]'的副本
Gjdx = idx;                  %列索引矩阵，大小为 nSmp × r，每行是对应的 Landmark 索引。
Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,p);%使用行索引、列索引和相似度值构建稀疏矩阵 Z
%% Graph decomposition图分解,将矩阵Z的每一列归一化
feaSum = full(sqrt(sum(Z,1)));%将Z矩阵每列求和再取平方根,稀疏矩阵转为稠密矩阵
feaSum = max(feaSum, 1e-12);  %对 feaSum 中的每个元素取最大值，确保其不小于 1e-12,避免出现除以零的情况。
Z = Z./feaSum(ones(size(Z,1),1),:);
Z=double(Z);%Z转换为双精度浮点数类型,这里Z是11741×500的。
Z = Z';   %这里Z是500×11741的适用于Fair SC进行SVD
fprintf('----------Z的大小为------------\n');
disp(size(Z));
%% 选取整个数据集的敏感属性Group
 Group = readmatrix('D:\ZLF\LFSC\databases\Bank_Sensitive.csv');
 Group =Group (1:10000,:);
%% 标准对称规范化的Fair SC
% 确保调用的是最新版本
[B,Q,N]=Fair_SC_normalized(Z,Group);
% 将归一化后的Z进行奇异值分解(SVD)得到V
[U,S,V]= mySVD(N,k+1);%对归一化后的 Z 进行奇异值分解（SVD）,提取前 k+1 个奇异向量
%normNFair=adj*B*Q逆,V=normNFair'*U*S逆

%% Z进行SVD后得到V。去掉V的第一个列（第一个奇异向量),通常对应常数特征，对聚类没有帮助.
V(:,1) = [];
 %% 将V归一化奇异向量
V=V./repmat(sqrt(sum(V.^2,2)),1,k);
%% 将V进行k-means得到聚类标签(标准对称规范化的Fair SC)
H=(B/Q)*V;%这里B/Q=B*Q逆，所以H=B*Q逆*normNFair'*U*S逆=B*Q逆*V，其中V=normNFair'*U*S逆
LFSC_normalized_clusterlabels=litekmeans(H,k,'MaxIter',maxIter,'Replicates',numRep);
%% 调用平衡性balance
fprintf('----------LFSC_normalized的结果为------------\n');
LFSC_normalized_Avg_Balance = LFSC_Balance(LFSC_normalized_clusterlabels, Group);
fprintf('----------LFSC_normalized_Avg_Balance 为------------\n');
disp(LFSC_normalized_Avg_Balance);

%% 调用得分情况Fraction
LFSC_normalized_Avg_Fraction = LFSC_Fraction(LFSC_normalized_clusterlabels,Group);
fprintf('----------LFSC_normalized_Avg_Fraction为------------\n');
disp(LFSC_normalized_Avg_Fraction);






