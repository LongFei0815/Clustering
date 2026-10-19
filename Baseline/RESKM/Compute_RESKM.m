function [RESKM_clusterlabels,RESKM_Avg_Balance,RESKM_Avg_Fraction,U_o,RESKM_center] = Compute_RESKM(data, k,opts)
 % configuration

%step 1: Select Anchors     %第一阶段锚点选择
    
    [anchors] = anchorselection(data, opts);  %在anchorselection.m中进行操作这里opt包含三个字段r=2、s=3600、a=600即最近的锚点数量、采样集、锚点集
    
%step 2: Construct Graph   %第二阶段锚点图构造
    
    [Z] = construction(data, anchors, opts); %Z进行归一化处理代表B的高度稀疏表示，
    clear anchors;  %删除当前变量锚点，重新释放
   
    %step 3: Compute the eignvector of anchor space 
    [ U_o,U_a] = RESKM_mySVD(Z,k);  
    %mySVD特征分解，其中Z是稀疏表示，K是集群数量，Uo是前top-k个对象特征向量、Ua是top-k个锚点特征向量
    
    clear Z; %%清除当前变量锚点，重新释放
    
    %step 4: Perform kmeans
    [~,RESKM_center]=litekmeans(U_a,k,'Replicates',10, 'MaxIter',500);
    % 关键修正：获取数据点的数量n（通过U_o的行数）
    n = size(U_o, 1);  % U_o是对象特征向量矩阵，行数=数据点数量
    bb = full(sum(RESKM_center.*RESKM_center,2)');%将稀疏中心矩阵center每行的总和成列，利用full转换为全矩阵给bb
    ab = full(U_o*RESKM_center');%将稀疏矩阵Uo聚类中心转为全矩阵给ab,因为Uo是top-k对象的特征向量，数量和对象一样多、这里聚类中心为10组，故为7291×10
    D = bb(ones(1,n),:) - 2*ab;%这里ones(1,n)表示产生1×n的全1矩阵，即1行n列,在bb矩阵中，然后-2乘ab矩阵得到度矩阵
    [~, RESKM_clusterlabels] = min(D, [], 2);
    clear U_a dump labels U_o
    %% 读取敏感属性 Group
    Group = readmatrix('E:\ZhongLongFei_Code\LFSC\databases\ElevatorWX2022_Sensitive.csv');
    Group =Group (1:3000,:);
    fprintf('----------RESKM的结果------------\n');
    %% 调用平衡性 ComputeBalance
    RESKM_Avg_Balance = RESKM_Balance(RESKM_clusterlabels, Group);
    fprintf('----------  RESKM_Avg_Balance为------------\n');
    disp(RESKM_Avg_Balance);

    %% 调用得分情况 ComputeFraction
    RESKM_Avg_Fraction = RESKM_Fraction(RESKM_clusterlabels, Group);
    fprintf('---------- RESKM_Avg_Fraction为------------\n');
    disp(RESKM_Avg_Fraction);
end