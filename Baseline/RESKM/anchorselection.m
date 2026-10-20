%锚点选择步骤
%通过抽样绘制分布的方式来确定，而不是面向集群所有成员。
%(1)通过不替换的随机抽样生成大小为S的子集，这里S∈对象数据集O（7291×256个）;
% (2)基于S，执行a-Means;
% (3)将得到的中心作为锚点a，

function [anchors] = anchorselection(data, opts) %fea和opts为输入的参数值                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
%This function is to select anchors  选择锚点

nSmp=size(data,1);    %将矩阵fea的行数返回给nSmp，根据fea为7291×256矩阵，得出将7291×1给nSmp
if (~exist('opts','var'))  %如果不存在opts的变量
   opts = [];  %那么存在则返回一个空的数组给opts存放字段
end
% the number of data objects to perform a-means 
s = 20; %选样本集3400个
if isfield(opts, 's') %如果opts中包含由s指定域，则返回opts字段的值给样本集s
    s = opts.s;
end
% the number of the anchors
a = 3;%聚类获得中心作为锚点
if isfield(opts, 'a')  %如果opts中包含由a指定域，则返回opts.a给锚点数量s
    a = opts.a;
end
% the number of iterations of anchor a-means.选取的锚点a迭代次数这里设置为3
kmMaxIter = 5;  %设定kmMaxIter为迭代次数3次
if isfield(opts,'kmMaxIter')     %如果opts中包含由kmMaxIter指定域，则返回opts字段值给迭代次数kmMaxIter
    kmMaxIter = opts.kmMaxIter;
end
% the number of times to repeat a-means, alaways one.重复a-means次数
kmNumRep = 1;  %重复a-means次数kmNumRep = 1次
if isfield(opts,'kmNumRep')   %如果opts中包含由kmNumRepr指定域，则返回opts.kmNumRep给迭代次数 kmNumRep
    kmNumRep = opts.kmNumRep;
end
indSmp = randperm(nSmp); 
[dump, anchors] = litekmeans(data(indSmp(1:s),:),a,'MaxIter', kmMaxIter,'Replicates',kmNumRep);

clear indSmp dump 
end
