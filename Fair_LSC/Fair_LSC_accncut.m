%%%1、函数定义
function [AC,MIhat,Avg_balance,Fraction]=Fair_LSC_accncut(feat, gnd)
%用于计算聚类结果的准确率AC和归一化互信息MIhat

%%%2、计算聚类数目
nCluster=length( unique( gnd ) ) ;%计算真实标签gnd中的唯一值数量,即聚类的数目nCluster
%nCluster=7 ;%计算真实标签gnd中的唯一值数量,即聚类的数目nCluster
disp(['k = nCluster现在的大小为:  ' num2str(nCluster)]);
%%%2、设置 LSC 算法的参数
opts.r = 5;                   %Landmark的数量
opts.p = 15;                  %随机选择的样本数量
opts.kmMaxIter = 5;           %K-means算法的最大迭代次数
opts.mode = 'kmeans';

%%%3、调用 FairLSC 算法进行聚类
%% 标准对称规范化Fair LSC
%[res,Avg_balance,Fraction] = Compute_Fair_LSC_normalized(feat, nCluster,opts);%调用FairSC_norm 算法对 Z 进行聚类,返回聚类结果 res
%% 非标准对称规范化Fair LSC
%[res,Avg_balance,Fraction] = Compute_FairLSC_unnormalized(feat, nCluster,opts);%调用FairSC_unnorm 算法对 Z 进行聚类,返回聚类结果 res
%% 消融实验LSC_normalized
[res,Avg_balance,Fraction] = Ablation_LSC_normalized(feat,nCluster,opts);%消融实验LSC_normalized
%% 消融实验LSC
%[res,Avg_balance,Fraction] = Ablation_LSC(feat,nCluster,opts);%消融实验LSC
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