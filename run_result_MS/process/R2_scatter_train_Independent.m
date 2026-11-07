close all
clear
clc
set(groot, 'DefaultAxesFontName', 'Times New Roman');

%% 测试集
%% Proposed Model&Proposed Model without EN&VMD
load ..\My_Informer_CO_EN_16r1c_VMD_8_test_result_inversed.mat
predicted_model1 = predict_y1_My_Informer;
load ..\My_Informer_CO_data_26r1c_test_result_inversed.mat
actual_values = truth_y1;
predicted_model2 = predict_y1_My_Informer;

% R平方
R_squared_model1 = 1 - sum((actual_values - predicted_model1).^2) / sum((actual_values - mean(actual_values)).^2);
R_squared_model2 = 1 - sum((actual_values - predicted_model2).^2) / sum((actual_values - mean(actual_values)).^2);

% 散点图
figure
scatter(actual_values, predicted_model1,80,'red', 'Marker','.', 'DisplayName', sprintf('Proposed Model(R^2 = %.4f)', R_squared_model1));
hold on;
scatter(actual_values, predicted_model2,10,'blue', 'Marker', 'o', 'DisplayName', sprintf('Informer(R^2 = %.4f)', R_squared_model2));

% 添加轴标签和图标题
xlabel('True value');
ylabel('Prediction results');
% title('R-square scatter plot of Five models');
% 显示图例
legend('Location', 'southeast');

% 添加一条45度斜线，用于比较
plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'Color', 'k', HandleVisibility='off');
% plot([min(actual_values), 0.985], [min(actual_values), 0.985], '-', 'Color', 'k', HandleVisibility='off');

xlim([min(actual_values), max(actual_values)]);
ylim([min(actual_values), max(actual_values)]);
% xlim([min(actual_values), 0.9842]);
% ylim([min(actual_values), 0.9842]);
% 调整图的显示
% axis equal;
grid on;

%% My-Informer&CNN-LSTM
load ..\My_Informer_wind_EN_5r1c_train_result_inversed.mat
actual_values = truth_y1;
load ..\cnn_lstm_train_result_inversed.mat

predicted_model1 = predict_y1_My_Informer;
predicted_model2 = predict_y1_cnn_lstm;

% R平方
R_squared_model1 = 1 - sum((actual_values - predicted_model1).^2) / sum((actual_values - mean(actual_values)).^2);
R_squared_model2 = 1 - sum((actual_values - predicted_model2).^2) / sum((actual_values - mean(actual_values)).^2);

% 散点图
figure
scatter(actual_values, predicted_model1,80,'red', 'Marker','.', 'DisplayName', sprintf('Proposed Model(R^2 = %.4f)', R_squared_model1));
hold on
scatter(actual_values, predicted_model2,10,'blue', 'Marker', 'o', 'DisplayName', sprintf('CNN-LSTM(R^2 = %.4f)', R_squared_model2));

% 添加轴标签和图标题
xlabel('True value');
ylabel('Prediction results');
% title('R-square scatter plot of Five models');

% 添加一条45度斜线，用于比较
plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'Color', 'k', HandleVisibility='off');
% plot([min(actual_values), 0.985], [min(actual_values), 0.985], '-', 'Color', 'k', HandleVisibility='off');

legend()
xlim([min(actual_values), max(actual_values)]);
ylim([min(actual_values), max(actual_values)]);
% xlim([min(actual_values), 0.9842]);
% ylim([min(actual_values), 0.9842]);
% 调整图的显示
% axis equal;
grid on;

%% My-Informer&BiLSTM
load ..\My_Informer_wind_EN_5r1c_train_result_inversed.mat
actual_values = truth_y1;
load ..\Bilstm_train_result_inversed.mat

predicted_model1 = predict_y1_My_Informer;
predicted_model2 = predict_y1_Bilstm;

% R平方
R_squared_model1 = 1 - sum((actual_values - predicted_model1).^2) / sum((actual_values - mean(actual_values)).^2);
R_squared_model2 = 1 - sum((actual_values - predicted_model2).^2) / sum((actual_values - mean(actual_values)).^2);

% 散点图
figure;
scatter(actual_values, predicted_model1,80,'red', 'Marker','.', 'DisplayName', sprintf('Proposed Model(R^2 = %.4f)', R_squared_model1));
hold on;
scatter(actual_values, predicted_model2,10,'blue', 'Marker', 'o', 'DisplayName', sprintf('BiLSTM(R^2 = %.4f)', R_squared_model2));

% 添加轴标签和图标题
xlabel('True value');
ylabel('Prediction results');
% title('R-square scatter plot of Five models');
% 显示图例
legend('Location', 'southeast');

% 添加一条45度斜线，用于比较
plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'Color', 'k', HandleVisibility='off');
% plot([min(actual_values), 0.985], [min(actual_values), 0.985], '-', 'Color', 'k', HandleVisibility='off');

xlim([min(actual_values), max(actual_values)]);
ylim([min(actual_values), max(actual_values)]);
% xlim([min(actual_values), 0.9842]);
% ylim([min(actual_values), 0.9842]);
% 调整图的显示
% axis equal;
grid on;

% %% 测试集
% load ..\My_Informer_wind_EN_5r1c_test_result_inversed.mat
% load ..\Informer_wind_EN_5r1c_test_result_inversed.mat
% load ..\cnn_lstm_test_result_inversed.mat
% load ..\Bilstm_test_result_inversed.mat
% 
% actual_values = truth_y1;
% predicted_model1 = predict_y1_My_Informer;
% predicted_model2 = predict_y1_Informer;
% predicted_model3 = predict_y1_cnn_lstm;
% predicted_model4 = predict_y1_Bilstm;
% 
% % R平方
% R_squared_model1 = 1 - sum((actual_values - predicted_model1).^2) / sum((actual_values - mean(actual_values)).^2);
% R_squared_model2 = 1 - sum((actual_values - predicted_model2).^2) / sum((actual_values - mean(actual_values)).^2);
% R_squared_model3 = 1 - sum((actual_values - predicted_model3).^2) / sum((actual_values - mean(actual_values)).^2);
% R_squared_model4 = 1 - sum((actual_values - predicted_model4).^2) / sum((actual_values - mean(actual_values)).^2);
% 
% % 散点图
% figure;
% scatter(actual_values, predicted_model1,300,'black', 'Marker','.', 'DisplayName', sprintf('Proposed Model(R^2 = %.4f)', R_squared_model1));
% hold on;
% scatter(actual_values, predicted_model2,80,'red', 'Marker', 'diamond', 'DisplayName', sprintf('Informer(R^2 = %.4f)', R_squared_model2));
% scatter(actual_values, predicted_model3,80,'green', 'Marker', 'o', 'DisplayName', sprintf('CNN-LSTM(R^2 = %.4f)', R_squared_model3));
% scatter(actual_values, predicted_model4,80,'blue', 'Marker', 'hexagram', 'DisplayName', sprintf('BiLSTM(R^2 = %.4f)', R_squared_model4));
% 
% % 添加轴标签和图标题
% xlabel('True value');
% ylabel('Prediction results');
% % title('R-square scatter plot of Five models');
% % 显示图例
% legend('Location', 'southeast');
% 
% % 添加一条45度斜线，用于比较
% plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'Color', 'k', HandleVisibility='off');
% % plot([min(actual_values), 0.985], [min(actual_values), 0.985], '-', 'Color', 'k', HandleVisibility='off');
% 
% xlim([min(actual_values), max(actual_values)]);
% ylim([min(actual_values), max(actual_values)]);
% % xlim([min(actual_values), 0.9842]);
% % ylim([min(actual_values), 0.9842]);
% % 调整图的显示
% % axis equal;
grid on;