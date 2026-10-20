function [Z] = construction(data, anchors, opts )  
%Z结构如下:输入高度稀疏表示矩阵B的归一化Z，这里B是稀疏表示矩阵，每一项都是对应锚点与数据对象之间的关联值。
%对于每个数据对象，B只记录最近的r个锚点,所以并具有高度稀疏性，它只保留了r个重要的信息。
%对B进一步归一化处理利用Da(以B的行和为项的对角矩阵)，Do(以B的列和为项的对角矩阵)，进行表示
%Z=Da -1/2 B Do-/2
%   graph constrcution

% Set and parse parameters
if (~exist('opts','var'))
   opts = [];
end
% the number of anchors 
a = 3;
if isfield(opts, 'a')
    a = opts.a;
end
% the number of neighbour anchors 最近的锚点数r
r = 2;
if isfield(opts,'r')
    r = opts.r;
end
nSmp=size(data,1); %将输入的数据data的行数返回给数据对象nSmp。

% Z construction 
   
D = EuDist2(data,anchors,0);
%data是一个数据集矩阵，每行代表一个样本，每一列代表一个特征，
%anchors锚点集，计算数据集到每个样本欧式距离。
%O是一个参数，指欧式距离计算方法
%最后EuDist2是计算数据集中每个样本到锚点的欧式距离，
%返回一个度矩阵D，D行数代表数据集中样本数，列数是锚点集的样本数，每个元素表示对应样本到对应锚点的欧式距离O
if isfield(opts,'sigma')  %如果opts中包含由sigma指定域，则返回opts字段的值给sigma
    sigma = opts.sigma;
else
    sigma = mean(mean(D));  %否则将对度矩阵求两次平均操作，
    % 第一次求平均是mean(D)对每一列进行求平均得到一个行向量。
    % 第二次mean(mean(D))求平均是对第一次求平均得到的行向量在进行求平均，得到一个标量(指数值大小)，即D的平均值
    %最后将得到的标量(D平均值)给sigma
end



% find the r nearest anchors
dump = zeros(nSmp,r);%一个由nSmp×r的全零矩阵给dump
idx = dump;%把dump赋值给idx

for i = 1:r %i从1到r
    [dump(:,i),idx(:,i)] = min(D,[],2);
    %D是二维度矩阵，[]空矩阵，2表示对矩阵行操作，min不仅可以返回每行最小值，还可以返回这些最小值得列索引 
    %min(D,[],2)是一个列向量，包含度矩阵D中每一行的最小值，得到一个列向量。
    %dump(:,i)表示用来存储找到的最小值，idx(:,i)表示用来存储招待的最小值所在的列索引，(:,i)表示操作矩阵的第i列，每次操作都会更新i
 %总结就是便利度矩阵D的每一行，找出每行的最小值存储在dump的第i列，对应的行中索引位置存储在idx的第i列，注意i是循环的。
  
   
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
    %前面idx(:,i)得到了D的每行最小值所在的列索引位置，idx(:,i)-1则把i列所有元素-1，为了将索引从1变为0，因为i是从1开始的。
    %(idx(:,i)-1)*nSmp将得到的值乘nSmp（nSmp=size(data,1);%将输入的数据data的行数返回给数据对象nSmp。为了将索引的值转为矩阵中的绝对位置。
    %[1:nSmp]'创建一个列向量，包含从1到nSmp的连续整数，因为前面nSmp就是一个数据data的行数。
    %最后相加得到一列连续整数的向量给temp存储了i从1到r列矩阵中的索引。
    D(temp) = 1e100; %将度矩阵D中的特定索引temp元素设置为1e100，为了将D中设一个非常大的值，在后续处理中忽略这些在，或者标记为异常值。
end
%这个整体就是在最近的锚点r中，从循环1：r次找到度矩阵D每行的最小值及其索引，将这些最小值设置为一个很大的值，下次计算时忽略掉，不会再选择先前选过的值，这样的话可以保证采集的最近锚点不会重复，保证了高度稀疏。

dump = exp(-dump/(2*sigma ));
%前面sigma是求出度矩阵D的平均值，那么dump是前面计算的度矩阵D每行最小值存储在dump的第i列中，对应公式中的Bij是第i个锚点和第j个数据对象的相似度。
Gsdx = dump;%将dump即Bij给Gsdx
Gidx = repmat([1:nSmp]',1,r);
%将列向量[1:nSmp]'从1到nSmp的连续整数，沿着列的方向重复r次，生成一个nSmp×r的矩阵Gidx
Gjdx = idx;%前面求过的idx矩阵给Gjdx
Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,a);
%Gidx(:),Gjdx(:),Gsdx(:)将矩阵展开为列向量，nSmp矩阵行数，a为列数，(Gidx(:),Gjdx(:),Gsdx(:))代表填入的矩阵值。
%最后将填入的稀疏矩阵给Z，因为稀疏矩阵计算降低复杂度，节约空间。


%normalize
D_a = 1./(sqrt(sum(Z,2))+10^(-6));%因为Da是B的行和项。
%sqrt(sum(Z,2)计算稀疏矩阵Z的每行元素和，得到一个列向量，进行开根号处理，再加上10的-6次方，最后取倒数。
D_o = 1./(sqrt(sum(Z,1))+10^(-6));%因为Da是B的列和项。
%sqrt(sum(Z,1)计算稀疏矩阵Z的每列元素和，得到一个行向量，进行开根号处理，再加上10的-6次方，最后取倒数。
Z = repmat(D_a,1,a).* Z .* repmat(D_o,nSmp, 1);
%repmat(D_a,1,a)将列向量Da在行的方向上复制a次(锚点)。
%repmat(D_o,nSmp, 1)将行向量Do在列的方向上复制nSmp次。
%repmat(D_a,1,a).* Z .* repmat(D_o,nSmp, 1)将得到的矩阵同Z相乘再乘Do
%这里在论文中Z=Da -1/2 B Do-1/2是锚定度Da与对象度Do进行归一化处理

end

