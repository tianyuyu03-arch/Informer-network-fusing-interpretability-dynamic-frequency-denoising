clear
clc
close all
% num返回的是excel中的数据，txt输出的是文本内容，raw输出的是未处理数据
x = csvread('CO_EN_16r1c.csv') ;

i = 0;
high_weight = 0;
train_size = 2900 + 256;
test_size = 100;
%% 第一组
% 参数设置
i = i + 1;
train_size = train_size + test_size;
% 拿出该组数据
data = x(1:train_size + test_size, :);
input = data(:, 1:16);
output = data(:, 17);
% 分割输入的训练和测试集
input_train = input(1:train_size, :);
input_test = input(train_size + 1:end, :);
% 分别进行VMD
input_train_vmd = vmd_lownoise(input_train, high_weight);
input_test_vmd = vmd_lownoise(input_test, high_weight);
input_new = [input_train_vmd;input_test_vmd];
% 重新组合该组数据
data_new = [input_new output];
file_name = ['weight' num2str(high_weight*10) '\CO_data_EN_16r1c_VMD_' num2str(high_weight*10) '_' num2str(i) '.csv'];
% file_name = 'weight3\CO_data_26r1c_EN_VMD_3_1.csv';
writematrix(data_new,file_name);

%% 第二组
% 参数设置
i = i + 1;
train_size = train_size + test_size;
% 拿出该组数据
data = x(1:train_size + test_size, :);
input = data(:, 1:16);
output = data(:, 17);
% 分割输入的训练和测试集
input_train = input(1:train_size, :);
input_test = input(train_size + 1:end, :);
% 分别进行VMD
input_train_vmd = vmd_lownoise(input_train, high_weight);
input_test_vmd = vmd_lownoise(input_test, high_weight);
input_new = [input_train_vmd;input_test_vmd];
% 重新组合该组数据
data_new = [input_new output];
file_name = ['weight' num2str(high_weight*10) '\CO_data_EN_16r1c_VMD_' num2str(high_weight*10) '_' num2str(i) '.csv'];
% file_name = 'weight3\CO_data_26r1c_EN_VMD_3_1.csv';
writematrix(data_new,file_name);

%% 第三组
% 参数设置
i = i + 1;
train_size = train_size + test_size;
% 拿出该组数据
data = x(1:train_size + test_size, :);
input = data(:, 1:16);
output = data(:, 17);
% 分割输入的训练和测试集
input_train = input(1:train_size, :);
input_test = input(train_size + 1:end, :);
% 分别进行VMD
input_train_vmd = vmd_lownoise(input_train, high_weight);
input_test_vmd = vmd_lownoise(input_test, high_weight);
input_new = [input_train_vmd;input_test_vmd];
% 重新组合该组数据
data_new = [input_new output];
file_name = ['weight' num2str(high_weight*10) '\CO_data_EN_16r1c_VMD_' num2str(high_weight*10) '_' num2str(i) '.csv'];
% file_name = 'weight3\CO_data_26r1c_EN_VMD_3_1.csv';
writematrix(data_new,file_name);

%% 第四组
% 参数设置
i = i + 1;
train_size = train_size + test_size;
% 拿出该组数据
data = x(1:train_size + test_size, :);
input = data(:, 1:16);
output = data(:, 17);
% 分割输入的训练和测试集
input_train = input(1:train_size, :);
input_test = input(train_size + 1:end, :);
% 分别进行VMD
input_train_vmd = vmd_lownoise(input_train, high_weight);
input_test_vmd = vmd_lownoise(input_test, high_weight);
input_new = [input_train_vmd;input_test_vmd];
% 重新组合该组数据
data_new = [input_new output];
file_name = ['weight' num2str(high_weight*10) '\CO_data_EN_16r1c_VMD_' num2str(high_weight*10) '_' num2str(i) '.csv'];
% file_name = 'weight3\CO_data_26r1c_EN_VMD_3_1.csv';
writematrix(data_new,file_name);

%% 第五组
% 参数设置
i = i + 1;
train_size = train_size + test_size;
% 拿出该组数据
data = x(1:train_size + test_size, :);
input = data(:, 1:16);
output = data(:, 17);
% 分割输入的训练和测试集
input_train = input(1:train_size, :);
input_test = input(train_size + 1:end, :);
% 分别进行VMD
input_train_vmd = vmd_lownoise(input_train, high_weight);
input_test_vmd = vmd_lownoise(input_test, high_weight);
input_new = [input_train_vmd;input_test_vmd];
% 重新组合该组数据
data_new = [input_new output];
file_name = ['weight' num2str(high_weight*10) '\CO_data_EN_16r1c_VMD_' num2str(high_weight*10) '_' num2str(i) '.csv'];
% file_name = 'weight3\CO_data_26r1c_EN_VMD_3_1.csv';
writematrix(data_new,file_name);

%% 第六组
% 参数设置
i = i + 1;
train_size = train_size + test_size;
% 拿出该组数据
data = x(1:train_size + test_size, :);
input = data(:, 1:16);
output = data(:, 17);
% 分割输入的训练和测试集
input_train = input(1:train_size, :);
input_test = input(train_size + 1:end, :);
% 分别进行VMD
input_train_vmd = vmd_lownoise(input_train, high_weight);
input_test_vmd = vmd_lownoise(input_test, high_weight);
input_new = [input_train_vmd;input_test_vmd];
% 重新组合该组数据
data_new = [input_new output];
file_name = ['weight' num2str(high_weight*10) '\CO_data_EN_16r1c_VMD_' num2str(high_weight*10) '_' num2str(i) '.csv'];
% file_name = 'weight3\CO_data_26r1c_EN_VMD_3_1.csv';
writematrix(data_new,file_name);

%% 第七组
% 参数设置
i = i + 1;
train_size = train_size + test_size;
% 拿出该组数据
data = x(1:train_size + test_size, :);
input = data(:, 1:16);
output = data(:, 17);
% 分割输入的训练和测试集
input_train = input(1:train_size, :);
input_test = input(train_size + 1:end, :);
% 分别进行VMD
input_train_vmd = vmd_lownoise(input_train, high_weight);
input_test_vmd = vmd_lownoise(input_test, high_weight);
input_new = [input_train_vmd;input_test_vmd];
% 重新组合该组数据
data_new = [input_new output];
file_name = ['weight' num2str(high_weight*10) '\CO_data_EN_16r1c_VMD_' num2str(high_weight*10) '_' num2str(i) '.csv'];
% file_name = 'weight3\CO_data_26r1c_EN_VMD_3_1.csv';
writematrix(data_new,file_name);

%% 第八组
% 参数设置
i = i + 1;
train_size = train_size + test_size;
% 拿出该组数据
data = x(1:train_size + test_size, :);
input = data(:, 1:16);
output = data(:, 17);
% 分割输入的训练和测试集
input_train = input(1:train_size, :);
input_test = input(train_size + 1:end, :);
% 分别进行VMD
input_train_vmd = vmd_lownoise(input_train, high_weight);
input_test_vmd = vmd_lownoise(input_test, high_weight);
input_new = [input_train_vmd;input_test_vmd];
% 重新组合该组数据
data_new = [input_new output];
file_name = ['weight' num2str(high_weight*10) '\CO_data_EN_16r1c_VMD_' num2str(high_weight*10) '_' num2str(i) '.csv'];
% file_name = 'weight3\CO_data_26r1c_EN_VMD_3_1.csv';
writematrix(data_new,file_name);

%% 第九组
% 参数设置
i = i + 1;
train_size = train_size + test_size;
% 拿出该组数据
data = x(1:train_size + test_size, :);
input = data(:, 1:16);
output = data(:, 17);
% 分割输入的训练和测试集
input_train = input(1:train_size, :);
input_test = input(train_size + 1:end, :);
% 分别进行VMD
input_train_vmd = vmd_lownoise(input_train, high_weight);
input_test_vmd = vmd_lownoise(input_test, high_weight);
input_new = [input_train_vmd;input_test_vmd];
% 重新组合该组数据
data_new = [input_new output];
file_name = ['weight' num2str(high_weight*10) '\CO_data_EN_16r1c_VMD_' num2str(high_weight*10) '_' num2str(i) '.csv'];
% file_name = 'weight3\CO_data_26r1c_EN_VMD_3_1.csv';
writematrix(data_new,file_name);

%% 第十组
% 参数设置
i = i + 1;
train_size = train_size + test_size;
% 拿出该组数据
data = x(1:train_size + test_size, :);
input = data(:, 1:16);
output = data(:, 17);
% 分割输入的训练和测试集
input_train = input(1:train_size, :);
input_test = input(train_size + 1:end, :);
% 分别进行VMD
input_train_vmd = vmd_lownoise(input_train, high_weight);
input_test_vmd = vmd_lownoise(input_test, high_weight);
input_new = [input_train_vmd;input_test_vmd];
% 重新组合该组数据
data_new = [input_new output];
file_name = ['weight' num2str(high_weight*10) '\CO_data_EN_16r1c_VMD_' num2str(high_weight*10) '_' num2str(i) '.csv'];
% file_name = 'weight3\CO_data_26r1c_EN_VMD_3_1.csv';
writematrix(data_new,file_name);