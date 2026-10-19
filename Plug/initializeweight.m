function weight=initializeweight(X,D,opts)
%输入代表点X(输入矩阵)和D(目标数据矩阵)、模型选项结构体opts。
%输出初始化后的神经网络权重weight
%% Initializing weights初始化神经网络的权重
d=size(X,1);%d是输入数据X的行数。
m=size(D,2);%m是目标数据D的列数。
hidnum=[d opts.hidnum m];%hidnum是一个向量,表示神经网络的每一层的节点数。
%输入层的节点数d=size(X,1)、opts.hidnum=[1500]是隐藏层神经元数量从rpcmdemo_mnist函数中获取、输出层的节点数m=size(D,2)
for ii=2:size(hidnum,2)  %%返回hidnum列数,一共有3列,ii从2到3。使用 for 循环遍历神经网络的每一层（从第二层开始）
weight{ii}=0.01*randn(hidnum(ii),hidnum(ii-1));
%对于ii每一层(有两层ii2和ii=3),hidnum(ii)是当前层的节点数,hidnum(ii-1)是前一层的节点数。
%0.01*randn(hidnum(ii), hidnum(ii-1)) 生成一个大小为 [hidnum(ii), hidnum(ii-1)]的随机矩阵进行缩小100倍,避免初始权重过大,元素服从标准正态分布。
end
end
%weight得到的是一个元胞数组,weight=[W1,W2]其中每个元素是一个权重矩阵。
%W1是输入层到隐藏层的权重、W2是隐藏层到输出层的权重。
%weight{2}=20×30的权重矩阵
%weight{3}=40×60的权重矩阵
%可见actfun函数中。