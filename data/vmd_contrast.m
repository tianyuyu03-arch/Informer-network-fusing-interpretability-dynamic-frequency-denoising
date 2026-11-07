clc
clear
close all

%% 训练集
%% 导入数据 signal，设置参数
K = 2;  % 模态数
alpha = 2000;  % 正则化参数
weight = 1;
file_name = 'data/CO_EN_16r1c_VMD_10.csv';

xx=csvread('data/CO_EN_16r1c.csv');
data = xx(2:end,:);
data_train = data(1:3000+256+1, :);
data_test = data(3000+256+1+1:end, :);

%% 运行 VMD
high_fqs = [];
low_fqs = [];
for i = 1:size(data_train, 2)-1
    real_u=data_train(:,i);
    [imf,residual] = vmd(real_u,'NumIMF', K,'PenaltyFactor',100);
    high_fq = imf(:, 1);
    low_fq = imf(:, 2);
    high_fqs = [high_fqs, high_fq];
    low_fqs = [low_fqs, low_fq];
end

%% 画高频
% figure(1)
% t1 = tiledlayout(4,4,TileSpacing="compact",Padding="compact");
% for n = 1:size(data_train, 2)-1
%     ax(n) = nexttile(t1);
%     plot(high_fqs(:,n)')
%     xlim([1 5000])
%     xlabel("Time (s)")
% end

%% 画低频
% figure(2)
% t2 = tiledlayout(4,4,TileSpacing="compact",Padding="compact");
% for n = 1:size(data_train, 2)-1
%     ax(n) = nexttile(t2);
%     plot(low_fqs(:,n)')
%     xlim([1 5000])
%     xlabel("Time (s)")
% end

%% 画重新组合后的数据
cleanX = low_fqs+weight*high_fqs ;

% for j = 1:size(data_train, 2)-1
%     figure(j+2)
%     plot(data_train(:, j));
%     hold on;
%     plot(cleanX(:, j))
%     legend("Original X","Clean X")
%     xlabel("Time (s)")
%     ylabel("Signal")
% end

data_new_train = [cleanX data_train(:, end)];


% close all

%% 测试集
%% 导入数据 signal，设置参数
K = 2;  % 模态数
alpha = 2000;  % 正则化参数
%% 运行 VMD
high_fqs = [];
low_fqs = [];
for i = 1:size(data_test, 2)-1
    real_u=data_test(:,i);
    [imf,residual] = vmd(real_u,'NumIMF', K,'PenaltyFactor',100);
    high_fq = imf(:, 1);
    low_fq = imf(:, 2);
    high_fqs = [high_fqs, high_fq];
    low_fqs = [low_fqs, low_fq];
end

%% 画高频
% figure;
% t1 = tiledlayout(4,4,TileSpacing="compact",Padding="compact");
% for n = 1:size(data_test, 2)-1
%     ax(n) = nexttile(t1);
%     plot(high_fqs(:,n)')
%     xlim([1 5000])
%     xlabel("Time (s)")
% end

%% 画低频
% figure;
% t2 = tiledlayout(4,4,TileSpacing="compact",Padding="compact");
% for n = 1:size(data_test, 2)-1
%     ax(n) = nexttile(t2);
%     plot(low_fqs(:,n)')
%     xlim([1 5000])
%     xlabel("Time (s)")
% end

%% 画重新组合后的数据
cleanX = low_fqs+weight*high_fqs ;

% for j = 1:size(data_test, 2)-1
%     figure(j+2)
%     plot(data_test(:, j));
%     hold on;
%     plot(cleanX(:, j))
%     legend("Original X","Clean X")
%     xlabel("Time (s)")
%     ylabel("Signal")
% end

data_new_test = [cleanX data_test(:, end)];

close all

data_new = [data_new_train; data_new_test];

% file_name = 'data/CO_EN_16r1c_VMD_103.csv';

writematrix(data_new,file_name);