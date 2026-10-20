function Fair_SC_normalized_clusterlabels= Fair_SC_normalized(W,k,sensitive)

%% INPUT:
%% adj...（加权）邻接矩阵的大小为n × n
%% k ...聚类数量
%% sensitive ...编码敏感属性的长度为n的向量
%% OUTPUT:
%
%% clusterLabels…长度为n的向量，包含每个数据点的聚类标签

n = size(W, 1);%获取相似性矩阵的行数
% converting sensitive to a vector with entries in [h] and building F %%%
%将敏感值转换为包含[h]项的向量，并构建F

%% 将敏感属性 sensitive 转换为一个整数编码的向量 sensitiveNEW，并为后续构建公平性约束矩阵 F 做准备
sens_unique=unique(sensitive);%提取敏感属性 sensitive 中的唯一值，例如sensitive=[A,B,A,C,D]',则sens_unique=[A,B,C,D]'
h = length(sens_unique);%获取敏感属性 sensitive 中的唯一值长度，即得到保护组h的类别数。
sens_unique=reshape(sens_unique,[1,h]);%将sens_unique转成一个1×h的行向量

sensitiveNEW=sensitive;%创建一个新的变量 sensitiveNEW，用于存储转换后的敏感属性

temp=1;%初始化一个临时变量 temp，用于整数编码
for ell=sens_unique %遍历 sens_unique 中的每个唯一值 ell
    sensitiveNEW(sensitive==ell)=temp;%将 sensitive 中等于 ell 的元素替换为 temp
    temp=temp+1;%更新 temp,使其递增
end
    %例如sens_unique =[A,B,C,D]'有4个唯一值,sensitive=[A,B,A,C,D]'。
    %则temp=1时，ell=A，得到sensitiveNEW=[1,B,1,C,D]'，同理temp=4时候得到sensitiveNEW=[1,2,1,3,4]'.
 %% 构造F=f(s)-|Vs|/n*1n矩阵是一个大小为n×(h-1)，用于确保聚类结果对敏感属性公平
F=zeros(n,h-1);%初始化公平性约束矩阵 F，大小为 n×(h−1)的零矩阵用来存储F

for ell=1:(h-1)%ell=1:(h-1)遍历敏感属性的前 h−1 个类别,最后一个类别不需要单独处理，因为可以通过前 h−1 个类别推导出来
    temp=(sensitiveNEW == ell);%表示 sensitiveNEW 中等于 ell 的元素(ell从1到h-1)提取出来,赋值给temp。
    F(temp,ell)=1; %令零矩阵F中第 ell 列中对应 temp 为 true 的位置设置为 1。表示当前数据点属于哪个保护组(看ell是第几列)
    groupSize = sum(temp);%计算出当前类别ell的大小（即数据点中属于该类别的数量）
    F(:,ell) = F(:,ell)-groupSize/n;%对 F 的第 ell 列进行中心化处理，减去该类别的全局比例。
end

degrees = sum(W, 1);
D = diag(degrees);
L = D-W;
Z = null(F');

Q=sqrtm(Z'*D*Z);

M=(Q\Z')*L*(Z/Q); 
M=(M+M')/2;

try
    [Y, eigValues] = eigs(M,k,'smallestabs','MaxIterations',500,'SubspaceDimension',min(size(M,1),max(2*k,25)));
catch
    [Y, eigValues] = eigs(M,k,'smallestreal','MaxIterations',1000,'SubspaceDimension',min(size(M,1),max(2*k,25)));
end

H = Z*(Q\Y);

Fair_SC_normalized_clusterlabels=litekmeans(H,k,'Replicates',10, 'MaxIter',500);
end




