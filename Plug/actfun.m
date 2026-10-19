function [H]=actfun(W,X,act_fun)
%用于计算神经网络的每一层的激活值
%输入W是weight神经网络权重、X输入矩阵、act_fun激活函数的类型,见下面。
%输出H包含每一层激活值的元胞数组。
H{1}=X;%将输入数据X赋值给 H{1},表示输入层的激活值。
n=numel(W);
%使用numel(W)获取权重元胞数组W的元素个数,即神经网络的层数（包括输入层、隐藏层和输出层）。

for i=2:n-1 %计算隐藏层激活值,从第 2 层到第 n-1 层
   switch act_fun %根据act_fun的值选择不同的激活函数：
          case 'sigm'%使用 sigmoid 激活函数。
                % Calculate the unit's outputs (including the bias term)
                H{i} = sigm(W{i}*H{i-1});
          case 'tanh'%使用双曲正切（tanh）激活函数。
                H{i} = tanh(W{i}*H{i-1});
          case 'max'%使用 ReLU（修正线性单元）激活函数。
                H{i} = max(0,W{i}*H{i-1});
          case 'shrink'%使用收缩激活函数（将绝对值小于theta的值置为零）。
                H1=W{i}*H{i-1};
                theta=0.1;
                H{i} =(((-theta<H1)+(H1<theta))<1.5).* H1;
          case 'softplus'%使用 softplus 激活函数。
                H{i} = log(1+exp(W{i}*H{i-1}));
          case 'linear'%使用线性激活函数（无激活函数）
                H{i} = W{i}*H{i-1};
   end
end
   H{n} = W{n}*H{n-1};%计算输出层的激活值,使用线性激活函数（无激活函数）
end
