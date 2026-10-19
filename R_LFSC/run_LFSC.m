clear;
addpath('D:\ZLF\LFSC\databases\');
addpath('D:\ZLF\LFSC\Plug\');
addpath('D:\ZLF\LFSC\R_LFSC\');
%% 加载数据
load Bank.mat;
feat = Bank_data(1:10000,:)';
gnd = Bank_labels(1:10000,:);
%% 参数设置
opts.epsilon = 0.01; % 学习率,决定模型参数更新步长。
opts.gamma = 0.001;  % 权重正则化参数,防止过拟合。
opts.act_fun = 'max';  % 激活函数,这里使用max。
opts.hidnum = 50;    % 隐藏层神经元数量,网络结构为 (输入层 → 1000个隐藏神经元 → 输出层)。
opts.NNmaxiter = 1;    % 每次迭代中训练神经网络的最大次数
opts.trainIter = 45;   % 训练总迭代次数。

% 定义4种范数模型
%model.set{1}='RPCM_f2';
%model.set{1}='RPCM_l1';
%model.set{3}='RPCM_unclear';
model.set{4}='RPCM_l1f2'; 

opts.overlinealpha = 1;     % 论文中的~α参数,控制模糊度
opts.display = 1;           % 打印每次迭代中的损失函数 
opts.choosenum = 20;        % 选择样本的数量(代表点X)用来训练RPCM，每个类别选择的样本
opts.choosestyle = 1;       % style==1  randomly select the selectnum samples 
                            %           from the data of every clustering center
                            % 从每个聚类中心的数据中随机抽取的样本，样本的每个类别都被选opts.choosenum次。
                            % style==0  randomly select the selectnum samples
                            %           from the whole data
                            % 从整个数据中随机抽取抽取的样本，就是opts.choosenum的值
opts.alpha=0.01;             %如果是RPCM_l1+F2模型，需要增加额外的参数opts.alpha,论文中的α
opts.beta = 0.01;         % 表示控制噪声影响β参数,单个值使用直接传递给RPCM使用。
                              
opts.theta = 0.005;        % 神经网络权重衰减参数
%opts.theta=[0.005, 0.001,0.0005,0.0001,0.00005,0]; 
opts.repeatnum = 1;             % 模型训练的运行次数
repeatnum = opts.repeatnum;     % 表示模型的重复运行次数,减少随机性，提高结果的稳定性, 多次会取平均值。
choosenum = opts.choosenum;     % 每次运行时从数据集中选择的样本数量（用于生成代表点 X），降低复杂度。每个类别选择的样本。
choosestyle = opts.choosestyle; % 样本选择方式（0表示从整个数据集中随机选择,1表示从每个聚类中心随机选择）
%betaset = 0.01;
%betaset = [0.001,0.01,0.1,1,10];  % 储存所有待测试的噪声β参数，用于循环内生效，其它的模型可能需要β参数。

%% 初始化变量（使用单元格数组存储Fraction）,给变量预先分配空间存储
num_runs = 10;
Total_ACC_details = zeros(repeatnum, num_runs);
Total_NMI_details = zeros(repeatnum, num_runs);
Total_Time_details = zeros(repeatnum, num_runs);
Total_Balance_details = zeros(repeatnum, num_runs);
Total_Fraction_details = cell(repeatnum, num_runs);  % 改为单元格数组
%% 在循环开始前添加初始化，给变量预先分配空间存储
Avg_ACC_runs = zeros(1, num_runs);
Avg_NMI_runs = zeros(1, num_runs);
Avg_Time_runs = zeros(1, num_runs); 
Avg_Balance_runs = zeros(1, num_runs);
time_elapsed = zeros(1, num_runs);
%% 外层运行10次
for run = 1:num_runs
    tic;
    opts.modelstyle = 'RPCM_l1f2';%固定住模型

    %% for imodel = 1:length(betaset)   %测试多个噪声β参数时候使用
        %opts.beta = betaset(imodel);%
        %fprintf('Model %s: beta=%.3f\n', opts.modelstyle, opts.beta
        %% 内层运行repeatnum次
        for ii = 1:repeatnum  
            [X, ~] = selectsrno(feat, gnd', choosenum, choosestyle);
            [acc, nmi, time, Balance, Fraction] = RPCM(X, feat, gnd, opts);
            disp(['beta = : ' num2str(opts.beta) '  repeatnum = :   ' num2str(ii)]);
            disp(['acc = : ' num2str(acc) '  nmi = : ' num2str(nmi)]);
            % 存储结果（Fraction直接存为向量）
            Total_ACC_details(ii, run) = acc;
            Total_NMI_details(ii, run) = nmi;
            Total_Time_details(ii, run) = time;
            Total_Balance_details(ii, run) = Balance;
            Total_Fraction_details{ii, run} = Fraction;  % 单元格存储
        end
    
    %% 计算当前外层 run 的内层 10 次平均值
    Avg_ACC_runs(run) = mean(Total_ACC_details(:, run));
    Avg_NMI_runs(run) = mean(Total_NMI_details(:, run));
    Avg_Time_runs(run) = mean(Total_Time_details(:, run));
    Avg_Balance_runs(run) = mean(Total_Balance_details(:, run));
    time_elapsed(run) = toc;
    fprintf('运行 %d: ACC = %.4f, NMI = %.4f, 时间 = %.4f 秒\n', run, acc, nmi, time_elapsed(run));
end

%% 后处理：计算Fraction各维度平均值
% 合并所有Fraction结果
all_fractions = cellfun(@(x) x(:).', Total_Fraction_details, 'UniformOutput', false);
all_fractions = vertcat(all_fractions{:});
num_fractions = size(all_fractions, 2);  % 自动确定分数数量

% 计算各维度平均值
final_Avg_fractions = mean(all_fractions, 1);

%% 计算其他指标的平均值
final_Avg_ACC = mean(Avg_ACC_runs);
final_Avg_NMI = mean(Avg_NMI_runs);
final_Avg_Time = mean(Avg_Time_runs);
final_Avg_Balance = mean(Avg_Balance_runs);


%% 输出结果
fprintf('\n最终平均值（基于 %d 次外层运行 × %d 次内层运行）：\n', num_runs, repeatnum);
fprintf('平均 ACC: %.4f\n', final_Avg_ACC);
fprintf('平均 NMI: %.4f\n', final_Avg_NMI);
fprintf('平均 Time: %.4f 秒\n', final_Avg_Time);
fprintf('平均 Balance: %.4f\n', final_Avg_Balance);

fprintf('\n各Fraction维度平均值：\n');
for f = 1:num_fractions
    fprintf('Fraction %d: %.4f\n', f, final_Avg_fractions(f));
end

%% 全部保存结果到工作区
assignin('base', 'RFLSC_ACC_details', Total_ACC_details);
assignin('base', 'RFLSC_NMI_details', Total_NMI_details);
assignin('base', 'RFLSC_Time_details', Total_Time_details);
assignin('base', 'RFLSC_Balance_details', Total_Balance_details);
assignin('base', 'RFLSC_Fraction_details', Total_Fraction_details);
assignin('base', 'RFLSC_Avg_fractions', final_Avg_fractions);