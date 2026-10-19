%%%1、函数定义
function [AC,MIhat,Avg_balance,Fraction]=RESKM_accncut(feat, gnd)
%用于计算聚类结果的准确率AC和归一化互信息MIhat
%这里opts是一个参数选项结构体，用来存储和解析命令行，USPS中三个字段，r、s、a
opts.r =3;     % the number of nearest anchors . 
opts.a = 20;    % the number of anchors.这里是S是m×s矩阵(S是数据对象的一部分)进行a-means后得到的聚类中心作为锚点a，为了降低成本
opts.s = 50;   % the number of sampled object to perform a-means anchor selection.通过不替换的随机抽样生成大小为S
opts.kmMaxIter = 5;           %K-means算法的最大迭代次数
%%%2、计算聚类数目
%nCluster=length( unique( gnd ) ) ;%计算真实标签gnd中的唯一值数量,即聚类的数目nCluster
nCluster=4 ;%计算真实标签gnd中的唯一值数量,即聚类的数目nCluster
disp(['k = nCluster现在的大小为:  ' num2str(nCluster)]);

%%%3、调用 SC 算法进行聚类
%% 标准化对称SC_normalized
[res,Avg_balance,Fraction] = Compute_RESKM(feat, nCluster,opts);%调用SC_normalized算法对 Z 进行聚类,返回聚类结果 res

%%%4、将聚类结果与真实标签对齐
res = bestMap(gnd,res);%使用 bestMap 函数将聚类结果 res 与真实标签 gnd 对齐,确保聚类标签与真实标签一致

%%%5. 计算聚类准确率AC
AC = length(find(gnd == res))/length(gnd);
%find(gnd == res)返回真实标签 gnd 和聚类结果 res 匹配的索引,
%length(find(gnd == res)) 是匹配的样本数量,length(gnd) 是总样本数量。
%AC是匹配样本数量与总样本数量的比值。

%%%6、计算归一化互信息（MIhat）
MIhat = MutualInfo(gnd,res);%调用MutualInfo函数计算
%调用MutualInfo函数计算特征真是标签gnd和聚类结果res之间的归一化互信息MIhat。