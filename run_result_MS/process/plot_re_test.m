close all
clear
clc

load ..\Informer_CO_data_26r1c_test_result_inversed.mat
predict_Informer = predict_y1_Informer(1, 1:1000);
load ..\My_Informer_CO_data_26r1c_test_result_inversed.mat
predict_MyInformer = predict_y1_My_Informer(1, 1:1000);
load ..\My_Informer_CO_EN_16r1c_VMD_8_test_result_inversed.mat
predict_VMD_EN_MyInformer = predict_y1_My_Informer;
load ..\cnn_lstm_test_result_inversed.mat
predict_CNN_LSTM = predict_y1_cnn_lstm;


%% Y1
% 示例数据：实际值和预测值
actual_values = truth_y1;

predicted_values = [predict_MyInformer; predict_Informer;predict_CNN_LSTM];

% 计算相对误差
relative_error(1,:) = abs(actual_values - predicted_values(1,:)) ./ abs(actual_values);
relative_error(2,:) = abs(actual_values - predicted_values(2,:)) ./ abs(actual_values);
relative_error(3,:) = abs(actual_values - predicted_values(3,:)) ./ abs(actual_values);
% relative_error(4,:) = abs(actual_values - predicted_values(4,:)) ./ abs(actual_values);

% 绘制相对误差的图表
figure;
% subplot(2,1,1)
set(groot, 'DefaultAxesFontName', 'Times New Roman'); % 设置默认字体

plot(relative_error(1,:), '-k', 'LineWidth', 1);
hold on;
plot(relative_error(2,:), '--r', 'LineWidth', 1);
hold on;
plot(relative_error(3,:), ':b', 'LineWidth', 1.2);
hold on;
% plot(relative_error(4,:), '--b', 'LineWidth', 1);
% hold on;
% plot(relative_error(5,:), '-c', 'LineWidth', 1);

xlabel('Sample');
ylabel('Relative error');
% title('Relative error of data based on different networks');
grid on;

% 显示图例
legend('Proposed Model','Informer','CNN-LSTM');

% 将Y轴百分比显示
ytickformat('percentage');

% 调整Y轴范围
ylim([0, 0.06]);
% xlim([0 length(relative_error)])

% %% Y2
% % 示例数据：实际值和预测值
% actual_values = truth_y2;
% 
% predicted_values = [predict_y2_ResLSTMNet_GAT; predict_y2_LSTM;predict_y2_GRU;predict_y2_CNN];
% 
% % 计算相对误差
% relative_error(1,:) = abs(actual_values - predicted_values(1,:)) ./ abs(actual_values);
% relative_error(2,:) = abs(actual_values - predicted_values(2,:)) ./ abs(actual_values);
% relative_error(3,:) = abs(actual_values - predicted_values(3,:)) ./ abs(actual_values);
% relative_error(4,:) = abs(actual_values - predicted_values(4,:)) ./ abs(actual_values);
% 
% % 绘制相对误差的图表
% % figure;
% subplot(2,1,2)
% 
% set(groot, 'DefaultAxesFontName', 'Times New Roman'); % 设置默认字体
% 
% plot(relative_error(1,:), '-k', 'LineWidth', 1);
% hold on;
% plot(relative_error(2,:), '--r', 'LineWidth', 1);
% hold on;
% plot(relative_error(3,:), ':g', 'LineWidth', 1);
% hold on;
% plot(relative_error(4,:), '--b', 'LineWidth', 1);
% hold on;
% % plot(relative_error(5,:), '-c', 'LineWidth', 1);
% 
% xlabel('Sampling point');
% ylabel('Output Y2 Relative error');
% % title('Relative error of data based on different networks');
% grid on;
% 
% % 显示图例
% % legend('ResLSTMNet-GAT','LSTM','GRU','CNN');
% 
% % 将Y轴百分比显示
% ytickformat('percentage');
% 
% % 调整Y轴范围
% ylim([0, 0.04]);
% xlim([0 length(relative_error)])
