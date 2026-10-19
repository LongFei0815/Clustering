function [acc,nmi,Avg_balance, Fraction]=LbSC_Update_acc_nmi(feat, gnd)

accplrall=0; %初始化accplrall(准确率)为0。
%% 得到AC给accplrall_temp，MIhat给nmiplrall_temp
[accplrall_temp, nmiplrall_temp,Avg_balance, Fraction]=LbSC_accncut(feat,gnd);
%调用accncutLSC函数,计算聚类结果的临时准确率accplrall_temp和NMInmiplrall_temp。

if accplrall_temp>=accplrall  %如果临时准确率高于当前最高准确率,则更新accplrall和nmiplrall。
   accplrall=accplrall_temp;
   nmiplrall=nmiplrall_temp;
end

if accplrall==0 %如果accplrall仍为0,说明未找到有效聚类结果，使用默认方法   
   [accplrall, nmiplrall]=accnmi( feat, gnd );%accnmi.m函数,计算聚类结果的ACC和NMI
end

%%返回结果
acc=accplrall;
acc=mean(acc);
nmi=nmiplrall;
nmi=mean(nmi);
end