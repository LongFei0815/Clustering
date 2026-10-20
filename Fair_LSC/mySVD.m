function [U,S,V] = mySVD(X,ReducedDim)
%mySVD    Accelerated singular value decomposition.加速奇异值分解。                                                 

%%   mySVD加速奇异值分解X∈R(m×n)
%%   [U,S,V] = mySVD(X)产生一个对角矩阵S,维度为X的秩,并以非负对角线元素按降序排列，而酉矩阵U和V使
%%   X = U*S*V':其中U∈R(m×m)和V∈R(n×n)的正交矩阵中的列向量分别为X的左右奇异值
%%              对角矩阵S∈R(m×n)对角线上的元素为X的奇异值。
%%   [U,S,V] = mySVD(X,ReducedDim)产生一个维度为ReducedDim(k+1)的对角矩阵S，
%%   其中的非负对角元素为递减阶，酉矩阵U和V，这样得到
%%   Xhat = U*S*V'是所有秩不超过 ReducedDim 的矩阵中对 X 的最佳逼近（以 F 范数衡量）。
%%   根据X的大小，mySVD计算X*X'或X'*X的特征向量,然后把它们转换成另一个向量的特征向量。

%% SVD具体步骤:0
%%    1、根据X的大小来选择mySVD是计算X'*X还是X*X'。
%%    2、第一种：当X∈行数nSmp＜mFea列数，计算X*X'∈R(nSmp×nSmp)＜X'*X∈R(mFea×mFea)
%%       第二种：当X∈行数nSmp＞mFea列数，计算X'*X∈R(nSmp×nSmp)＜X*X'∈R(mFea×mFea)
%%    3、求第一种X∈行数nSmp＜mFea列数，计算X*X'特征向量是U(加速),X'*X特征向量是V(整体)，得到V=X'U×S逆
%%       求第二种X∈行数nSmp＞mFea列数，计算X'*X特征向量是V(加速),X*X'特征向量是U(整体)，得到U=XV×S逆
%%    4、构造对角矩阵S得到奇异值:将X'*X和X*X'的特征值开平方,得到奇异值，并按照从大到小排列,构成对角矩阵S
%%    5、验证分解:确认U和V是正交矩阵,即U'U=I和V'V=I,验证X=USV'。
%%    根据矩阵X的大小nSmp×mFea选择计算X'*X还是X*X'，注意我们这里的行数nSmp＞mFea列数的。
%%    如果X矩阵中列数mFea/nSmp行数 ＞ 1.0713，说明行数使用X*X'∈mFea×mFea＜nSmp来加速
%%        X矩阵中列数mFea/nSmp行数 ＜ 1.0713，使用X'*X∈mFea×mFea来加速

MAX_MATRIX_SIZE = 1000; % You can change this number according your machine computational power
EIGVECTOR_RATIO = 0.1; % You can change this number according your machine computational power

if ~exist('ReducedDim','var') %如果未提供 ReducedDim，则默认设置为 0，表示不进行降维
    ReducedDim = 0;
end %已经提供了ReducedDim的值是k+1,即整个数据集Y聚类簇的类别是[1~4]共4个+1
%% %为了符合论文对X进行转置变成行数nSmp＜列数mFea的类型，使用X=X'。
[nSmp, mFea] = size(X);         %nSmp=500行数、mFea=11741列数.

%% 如果X是行数n＜m列数，使用X*X'∈n×n＜m×m进行加速，最后是V=X'U×Σ逆
if mFea/nSmp > 1.0713           
    ddata = X*X';               %所以使用X*X'([500×11741]×[11741×500]=500×500)矩阵进行加速。
    ddata = max(ddata,ddata');%确保ddata是一个对称矩阵
    dimMatrix = size(ddata,1);
%% 计算特征向量，为了和其他对比算法保持一致，使用eigs函数，但是我们求的是前k个最大特征值的特征向量
     if (ReducedDim > 0) && (dimMatrix > MAX_MATRIX_SIZE) && (ReducedDim < dimMatrix*EIGVECTOR_RATIO)
        option = struct('disp',0);
        [U, eigvalue] = eigs(ddata,ReducedDim,'la',option);
        eigvalue = diag(eigvalue);
    else
        if issparse(ddata)
            ddata = full(ddata);
        end
        
        [U, eigvalue] = eig(ddata);
        eigvalue = diag(eigvalue);
        [dump, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        U = U(:, index);
    end
    clear ddata;
%if ReducedDim > 0
        %k = ReducedDim;
        %try
            %[U, eigvalue] = eigs(ddata, k, 'largestabs', 'MaxIterations', 500, 'SubspaceDimension', min(size(ddata,1), max(2*k, 25)));
        %catch
            %[U, eigvalue] = eigs(ddata, k, 'largestreal', 'MaxIterations', 1000, 'SubspaceDimension', min(size(ddata,1), max(2*k, 25)));
        %end
        %eigvalue = diag(eigvalue);
    %else
      
    %end

    %clear ddata;
 %% 特征值 eigvalue 和特征向量矩阵 U 进行进一步处理，并计算奇异值分解（SVD） 
    %%计算特征值 eigvalue 的最大绝对值,后续归一化特征值，过滤掉非常小的特征值。
    maxEigValue = max(abs(eigvalue));
    %%找到eigvalue特征值中相对较小的值小于最大特征值的1e-10倍，通常被认为是数值误差或噪声，可以忽略。
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-10);
    %%删除 eigvalue 中较小的特征值(误差或噪声)和对应的特征向量U
    eigvalue(eigIdx) = [];
    U(:,eigIdx) = [];
 %% 根据ReducedDim（即k+1)的值对特征值 eigvalue 和特征向量矩阵 U 进行截断，主要用于降维作用。
 %% 如果 ReducedDim 有效且小于当前特征值eigvalue的数量，则保留前 ReducedDim 个特征值和对应的特征向量。
    if (ReducedDim > 0) && (ReducedDim < length(eigvalue))%ReducedDim > 0且小于当前特征值eigvalue的数量
        eigvalue = eigvalue(1:ReducedDim);%保留前 ReducedDim 个特征值。
        U = U(:,1:ReducedDim);            %保留前 ReducedDim 列特征向量U
    end
 %% 计算奇异值分解（SVD）的最终结果，包括奇异值矩阵 S 和右奇异向量矩阵V ，得V=X'US逆
    %% 对特征值eigvalue列向量中每个元素取平方根,得到奇异值。因为奇异值是特征值的平方根
    eigvalue_Half = eigvalue.^.5;
    S =  spdiags(eigvalue_Half,0,length(eigvalue_Half),length(eigvalue_Half));%构造奇异值对角矩阵S
    %% 现在知道ddata=XX'的特征向量U和对角矩阵S后，自然得到V=X'US逆
    if nargout >= 3%mySVD是输出三个函数U,S,V。但是判断调用函数[U,S,V] = mySVD(Z,k+1);只。故为nargout=3
        eigvalue_MinusHalf = eigvalue_Half.^-1;%计算奇异值S的倒数,因为逆就是求倒数
        V = X'*(U.*repmat(eigvalue_MinusHalf',size(U,1),1));%得到V=X'US逆
     
    end
 %% 如果X是行数n＞m列数，使用X'*X∈m×m＜n×n进行加速，最后是U=XV×Σ逆
else          
    ddata = X'*X;
    ddata = max(ddata,ddata');
    
%% 计算特征向量，为了和其他对比算法保持一致，使用eigs函数，但是我们求的是前k个最大特征值的特征向量
    if ReducedDim > 0
        k = ReducedDim;
        try
            [V, eigvalue] = eigs(ddata, k, 'largestabs', 'MaxIterations', 500, 'SubspaceDimension', min(size(ddata,1), max(2*k, 25)));
        catch
            [V, eigvalue] = eigs(ddata, k, 'largestreal', 'MaxIterations', 1000, 'SubspaceDimension', min(size(ddata,1), max(2*k, 25)));
        end
        eigvalue = diag(eigvalue);
    else
      
    end

    clear ddata;
    
    maxEigValue = max(abs(eigvalue));
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-10);
    eigvalue(eigIdx) = [];
    V(:,eigIdx) = [];
    
    if (ReducedDim > 0) && (ReducedDim < length(eigvalue))
        eigvalue = eigvalue(1:ReducedDim);
        V = V(:,1:ReducedDim);
    end
    
    eigvalue_Half = eigvalue.^.5;
    S =  spdiags(eigvalue_Half,0,length(eigvalue_Half),length(eigvalue_Half));
    
    eigvalue_MinusHalf = eigvalue_Half.^-1;
    U = X*(V.*repmat(eigvalue_MinusHalf',size(V,1),1));
end







