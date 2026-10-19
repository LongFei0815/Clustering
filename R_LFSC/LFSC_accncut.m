%%%1、函数定义
function [AC,MIhat,Avg_balance, Fraction]=LFSC_accncut(Z, gnd)
%用于计算聚类结果的准确率AC和归一化互信息MIhat
%R=R2.*(R2>theta(i))+R2.*(R2<-theta(i)),Z是R'传递给的实际参数。gnd是特征标签

%%%2、设置 LSC 算法的参数
opt.r = 3;                   %Landmark的数量
opt.p = 20;                  %随机选择的样本数量
opt.kmMaxIter = 5;           %K-means算法的最大迭代次数
opt.mode = 'kmeans'; 
%意思数据集有1000个样本，opt.p =500做为随机选择样本数量做为landmark的候选集,再从500个候选样本选择3个样本做为landmark来近似计算相似度矩阵。
%opt.mode='random';根据opts.choosestyle = 1代表k-means或者0代表random

%%%3、计算聚类数目
nCluster=length( unique( gnd ) ) ;%计算真实标签gnd中的唯一值数量,即聚类的数目nCluster
%nCluster=2;
disp(['k = nCluster现在的大小为:  ' num2str(nCluster)]);
%%%4、数据预处理（注释部分)
% numtest = size(Z,1) ;
% for i = 1 : numtest
%    Z(i,:) = Z(i,:)./ norm(Z(i,:)) ; 
% end
% Z(~isnan(Z))=0;
%rand('twister',5489) 

%%%5、调用 LSC 算法进行聚类
%% 调用 LFSC_normalized 函数
[res, Avg_balance, Fraction]  = Compute_LFSC_normalized(Z, nCluster, opt);%调用RPCMFair_LSC_normalized算法对 Z 进行聚类,返回聚类结果 res
%% 调用 LFSC_unnormalized 函数
%[res, Avg_balance, Fraction] = Compute_LFSC_unnormalized(Z, nCluster, opt);%调用RPCMFair_LSC_unnormalized算法对 Z 进行聚类,返回聚类结果 res
%% 调用 LSC_normalized 函数
%[res, Avg_balance, Fraction] = Compute_LSC_normalized(Z, nCluster, opt);%调用RPCM_LSC_normalized算法对 Z 进行聚类,返回聚类结果 res

%%%6、将聚类结果与真实标签对齐
res = bestMap(gnd,res);%使用 bestMap 函数将聚类结果 res 与真实标签 gnd 对齐,确保聚类标签与真实标签一致

%%%7. 计算聚类准确率AC
AC = length(find(gnd == res))/length(gnd);
%find(gnd == res)返回真实标签 gnd 和聚类结果 res 匹配的索引,
%length(find(gnd == res)) 是匹配的样本数量,length(gnd) 是总样本数量。
%AC是匹配样本数量与总样本数量的比值。

%%%8、计算归一化互信息（MIhat）
MIhat = MutualInfo(gnd,res);%调用MutualInfo函数计算
%调用MutualInfo函数计算特征真是标签gnd和聚类结果res之间的归一化互信息MIhat。