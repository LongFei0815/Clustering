function D = EuDist2(fea_a,fea_b,bSqrt)
%% 输入参数:fea_a：矩阵 A，大小为 nSample_a × nFeature
%%         fea_b：矩阵 B，大小为 nSample_b × nFeature
%%         bSqrt：是否对距离矩阵取平方根（可选，默认为 1）
%% 输出参数:D支持两种模式→计算单个矩阵的欧氏距离 nSample_a × nSample_a 
%%                       计算两个矩阵之间欧氏距离 nSample_a × nSample_b
%EUDIST2 Efficiently Compute the Euclidean Distance Matrix by Exploring the
%通过探索欧几里得距离矩阵有效地计算
%Matlab matrix operations.
%
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b
%
%    Examples:
%
%       a = rand(500,10);
%       b = rand(1000,10);
%
%       A = EuDist2(a); % A: 500*500
%       D = EuDist2(a,b); % D: 500*1000
%
%   version 2.1 --November/2011
%   version 2.0 --May/2009
%   version 1.0 --November/2005
%
%   Written by Deng Cai (dengcai AT gmail.com)

%% 第一种模式：计算单个矩阵fea_a的欧式距离矩阵D=nSample_a × nSample_a 
if ~exist('bSqrt','var') %如果未提供 bSqrt 参数,则默认值为 1（表示对距离矩阵取平方根）
    bSqrt = 1;           %但是我们在LSC函数中提供参数0传递给bSqrt,则表示对距离矩阵不取平方根。
end

if (~exist('fea_b','var')) || isempty(fea_b) %如果未提供 fea_b 或 fea_b 为空,则计算矩阵 fea_a 自身的欧氏距离矩阵
  %% 计算fea_a矩阵每行样本的平方和相加，得到一个列向量的矩阵aa大小为nSample_a × 1, 
    aa = sum(fea_a.*fea_a,2); 
  %% 得到fea_a矩阵样本之间的内积,ab是nSample_a × nSample_a 的矩阵
    ab = fea_a*fea_a';        
  %% 如果 aa 是稀疏矩阵,则将其转换为稠密矩阵
    if issparse(aa)          
        aa = full(aa);        %稀疏矩阵在某些操作中可能效率较低，转换为稠密矩阵可以提高后续计算效率
    end
  %% 计算fea_a的欧式距离矩阵D  
    D = bsxfun(@plus,aa,aa') - 2*ab; %使用公式 D = aa + aa' - 2*ab 计算欧氏距离矩阵
    D(D<0) = 0;                      %将距离矩阵 D 中的负值置为零。由于数值误差,负数无意义
    if bSqrt                         %如果 bSqrt 为 1，则对距离矩阵 D 取平方根
        D = sqrt(D);
    end
  %% 确保距离矩阵D对称
    D = max(D,D'); %确保距离矩阵 D 是对称的。由于数值误差，D 可能不完全对称，因此取 D 和 D' 的最大值

%% 第二种模式：计算两个矩阵之间的欧式矩阵矩阵D=nSample_a × nSample_b 
else %如果提供了 fea_b，则计算两个矩阵 fea_a 和 fea_b 之间的欧氏距离矩阵
  %% 计算fea_a矩阵每行样本的平方和相加，得到一个列向量的矩阵aa大小为nSample_a × 1, 
    aa = sum(fea_a.*fea_a,2); 
  %% 计算fea_b矩阵每行样本的平方和相加，得到一个列向量的矩阵ab大小为nSample_b × 1, 
    bb = sum(fea_b.*fea_b,2); 
  %% 得到矩阵fea_a和矩阵fea_b之间样本的内积,ab 是nSample_a × nSample_b 的矩阵
    ab = fea_a*fea_b';
  %% 如果 aa或bb 是稀疏矩阵,则将其转换为稠密矩阵
    if issparse(aa)
        aa = full(aa);
        bb = full(bb);
    end
  %% 计算fea_a和fea_b的欧式距离矩阵D=nSample_a × nSample_b 的矩阵  
    D = bsxfun(@plus,aa,bb') - 2*ab;%使用公式 D = aa + bb' - 2*ab 计算欧氏距离矩阵
    D(D<0) = 0;                     %将距离矩阵 D 中的负值置为零。由于数值误差,负数无意义
    if bSqrt                        %如果 bSqrt 为 1，则对距离矩阵 D 取平方根
        D = sqrt(D);
    end
end
