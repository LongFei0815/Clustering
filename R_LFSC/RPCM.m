function [acc,nmi,time,Avg_balance, Fraction]=RPCM(X, feat, gnd, opts)
%%代码实现了一个基于神经网络的聚类算法RPCM,并通过计算聚类结果ACC和NMI来评估模型性能。
%%开始计时
tic;
%% 初始化模型参数
modelstyle=opts.modelstyle;  %当前使用的rpcm模型
theta=opts.theta;%从opts中获取神经网络的权重参数
D=X; %将在feat特征数据中选择的代表点X赋值给D，用于后续计算。
%% 初始化神经网络权重
weight=initializeweight(X,D,opts); %调用initializeweight.m函数。
%% 据模型类型选择不同的 RPCM 模型
switch modelstyle    
    %case 'RPCM_f2'
         %[weight, ~, ~] = RPCM_f2(X,D,weight,opts);
    %case 'RPCM_l1'
         %[weight, ~, ~, ~] = RPCM_l1(X,D,weight,opts);
    %case 'RPCM_unclear'
         %theta=[0.00001,0]; 
         %[weight, ~, ~, ~] = RPCM_unclear(X,D,weight,opts);
    case 'RPCM_l1f2'
         [weight, ~, ~, ~] = RPCM_l1f2(X,D,weight,opts);       
end

%Q = orth(A');
%B = A*Q;
%[Z,E] = preinexact_alm_lrr_l21(X,B,opts);
%Z = Q*Z;
%% Coding by fast computing the neural networks 计算神经网络激活值
H=actfun(weight,feat,opts.act_fun);%调用actfun.m函数，计算神经网络的激活值H。
R2=H{end}; %R2是H的最后一层输出，表示神经网络的最终表示。
%% Clustering by lbsc 通过landmark SC进行聚类

accplrall=0; %初始化accplrall(准确率)为0。
for i=1:size(theta,2)  %遍历theta的每个值,一共有6个值，i从1到6
%disp(['The truncated threshold:  ' num2str(theta(i))]);
%R = Shrink_L1(R2,theta(i));
if max(max(abs(R2)))>theta(i) %如果矩阵R2所有元素的绝对值中最大值大于当前神经网络权重theta(i)
R=R2.*(R2>theta(i))+R2.*(R2<-theta(i));
%导出到右侧工作区assignin('base', 'R', R);
%R2中的值大于神经网络权重theta(i),赋值1，反之0。再用R2的元素与一个1和0的矩阵相乘，其实就是保留R2中大于theta(i)的元素值。
%R2中的值小于神经网络权重theta(i),赋值1，反之0。再用R2的元素与一个1和0的矩阵相乘，其实就是保留R2中小于theta(i)的元素值。
%其实是对矩阵R2进行阈值处理，保留绝对值大于theta(i)的元素，其余元素置为零,得到矩阵R。
%判断R2的最大绝对值是否大于某个阈值 theta(i)


%% 调用accncutLSC函数[AC,MIhat]=accncutLSC(Z, gnd)。将R'传递给Z,gnd传递给gnd。
%% 得到AC给accplrall_temp，MIhat给nmiplrall_temp
[accplrall_temp, nmiplrall_temp,Avg_balance, Fraction]=LFSC_accncut(R',gnd);
%LFSC_accncut,计算聚类结果的临时准确率accplrall_temp和NMInmiplrall_temp。
end
if accplrall_temp>=accplrall  %如果临时准确率高于当前最高准确率,则更新accplrall和nmiplrall。
   accplrall=accplrall_temp;
   nmiplrall=nmiplrall_temp;
end
end
if accplrall==0 %如果accplrall仍为0,说明未找到有效聚类结果，使用默认方法
   R = ( abs(R2) + abs(R2') ) / 2 ; %对矩阵R2进行对称化处理，生成一个对称矩阵R(全是正值)
   [accplrall, nmiplrall]=accnmi( R, gnd );%accnmi.m函数,计算聚类结果的ACC和NMI
end

%%结束计时并记录时间
codingtime = toc;

%%显示结果
disp(['coding time:   ' num2str(codingtime)]);

%%返回结果
acc=accplrall;
acc=mean(acc);
nmi=nmiplrall;
nmi=mean(nmi);
time = codingtime;
time = mean(time);

end