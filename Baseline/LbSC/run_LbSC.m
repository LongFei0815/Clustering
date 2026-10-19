clear;

addpath('E:\ZhongLongFei_Code\LFSC\databases\');
addpath('E:\ZhongLongFei_Code\LFSC\Plug\');
addpath('E:\ZhongLongFei_Code\LFSC\Baseline\LbSC\');

%% 加载数据
load ElevatorWX2022.mat;
feat = ElevatorWX2022_data(1:3000,:);
gnd = ElevatorWX2022_labels(1:3000,:);
% 计算聚类的数目
k = length(unique(gnd)); % 计算真实标签 gnd 中的唯一值数量，即聚类的数目 nCluster
disp(['k= nCluster 原始的大小为: ', num2str(k)]);

% 初始化变量
num_runs = 10; % 运行次数
acc_sum = 0; % 用于累加 ACC
nmi_sum = 0; % 用于累加 NMI
time_sum = 0; % 用于累加运行时间
run_metrics.ACC = zeros(num_runs, 1);      % 存储每次运行的ACC
run_metrics.NMI = zeros(num_runs, 1);      % 存储每次运行的NMI
run_metrics.Time  = zeros(num_runs, 1);    % 存储每次运行的时间
run_metrics.Balance = zeros(num_runs, 1);  % 存储每次运行的平衡性
run_metrics.Fraction = cell(num_runs, 1);  % 使用元胞数组存储每次运行的得分

% 运行 10 次Fair SC
for run = 1:num_runs

    tic; % 开始计时
    [acc, nmi,Balance,Fraction] = LbSC_Update_acc_nmi(feat, gnd); % 调用谱聚类函数
    time_elapsed = toc; % 结束计时

    % 累加结果
    acc_sum = acc_sum + acc;
    nmi_sum = nmi_sum + nmi;
    time_sum = time_sum + time_elapsed;

    % 存储所有指标
    run_metrics.ACC(run) = acc;
    run_metrics.NMI(run) = nmi;
    run_metrics.Time(run) = time_elapsed;    
    run_metrics.Balance(run) = Balance;  
    run_metrics.Fraction{run} = Fraction;    

    % 打印当前运行的结果
    fprintf('运行 %d: ACC = %.4f, NMI = %.4f, 时间 = %.4f 秒\n', run, acc, nmi, time_elapsed);
end

% 计算平均值
Avg_acc = acc_sum / num_runs;
Avg_nmi = nmi_sum / num_runs;
Avg_time = time_sum / num_runs;
Avg_Balance = mean(run_metrics.Balance);  % 直接对存储的数组求平均
% 正确计算得分平均值（假设Fraction是向量）
all_Fractions = horzcat(run_metrics.Fraction{:}); 
Avg_Fraction = mean(all_Fractions, 2);

% 输出平均结果
fprintf('平均 ACC: %.4f\n', Avg_acc);
fprintf('平均 NMI: %.4f\n', Avg_nmi);
fprintf('平均 Time: %.4f 秒\n', Avg_time);
fprintf('平均 Balance: %.4f\n', Avg_Balance);
fprintf('平均 Fraction: %.4f\n', Avg_Fraction);
%% SC_norm 将关键变量显式保存到工作区
assignin('base', 'LbSC_ACCRecords', run_metrics.ACC);
assignin('base', 'LbSC_NMIRecords', run_metrics.NMI);
assignin('base', 'LbSC_TimeRecords', run_metrics.Time);
assignin('base', 'LbSC_BalanceRecords', run_metrics.Balance);
assignin('base', 'LbSC_FractionRecords', all_Fractions);

