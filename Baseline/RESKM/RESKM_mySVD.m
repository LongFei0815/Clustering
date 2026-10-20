function [C, U1] = RESKM_mySVD(Z, k) 
% this funciton is to solve the eigen problem
MAX_MATRIX_SIZE = 1600; % You can change this number according your machine computational power
EIGVECTOR_RATIO = 0.1; % You can change this number according your machine computational power
ReducedDim = k +1 ;
ddata = Z'*Z;%对应论文中Wo对象关系图
ddata = max(ddata,ddata');%将data和data转置每个元素比较，将两者中对应元素的较大值作为新的矩阵ddata的元素。
dimMatrix = size(ddata,1);%把ddata的行数赋给dimMatrix
 if (ReducedDim > 0) && (dimMatrix > MAX_MATRIX_SIZE) && (ReducedDim < dimMatrix*EIGVECTOR_RATIO)
     %如果(K+1)>0且(ddata行数>1600)且(K+1)<(ddata行数*0.1)，注意这里ddata行数的计算方式，不是原始的数据
    option = struct('disp',0);%创建一个option的结构体，其中的字段名disp的值设置为0
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
  maxEigValue = max(abs(eigvalue));
eigIdx = find(abs(eigvalue)/maxEigValue < 1e-10);
eigvalue(eigIdx) = [];
U(:,eigIdx) = [];

if (ReducedDim > 0) && (ReducedDim < length(eigvalue))
    eigvalue = eigvalue(1:ReducedDim);
    U = U(:,1:ReducedDim);
end

eigvalue_Half = eigvalue.^.5;
eigvalue_MinusHalf = eigvalue_Half.^-1;

U(:,1) = [];
eigvalue_MinusHalf(1) = [];
U1 = U.*repmat(eigvalue_MinusHalf',size(U,1),1);
C = Z * U1;
C = C ./repmat(sqrt(sum(C.^2,2)),1,k);
U1 = U1 ./repmat(sqrt(sum(U1.^2,2)),1,k);
