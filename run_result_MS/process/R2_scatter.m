close all
clear
clc

load ..\Informer_CO_data_26r1c_test_result_inversed.mat
predict_Informer = predict_y1_Informer;
load ..\My_Informer_CO_data_26r1c_test_result_inversed.mat
predict_MyInformer = predict_y1_My_Informer;
load ..\My_Informer_CO_EN_16r1c_VMD_8_test_result_inversed.mat
predict_VMD_EN_MyInformer = predict_y1_My_Informer;
load ..\cnn_lstm_test_result_inversed.mat
predict_CNN_LSTM = predict_y1_cnn_lstm;

set(groot, 'DefaultAxesFontName', 'Times New Roman');
%% Y1
actual_values = truth_y1;
predicted_model1 = predict_VMD_EN_MyInformer;
predicted_model2 = predict_MyInformer;
predicted_model3 = predict_Informer;
predicted_model4 = predict_CNN_LSTM;

% R平方
R_squared_model1 = 1 - sum((actual_values - predicted_model1).^2) / sum((actual_values - mean(actual_values)).^2);
R_squared_model2 = 1 - sum((actual_values - predicted_model2).^2) / sum((actual_values - mean(actual_values)).^2);
R_squared_model3 = 1 - sum((actual_values - predicted_model3).^2) / sum((actual_values - mean(actual_values)).^2);
R_squared_model4 = 1 - sum((actual_values - predicted_model4).^2) / sum((actual_values - mean(actual_values)).^2);
% 散点图
figure;
scatter(actual_values, predicted_model1,300,'black', 'Marker','.', 'DisplayName', sprintf('VMD-EN-ProposedModel(R^2 = %.4f)', R_squared_model1));
hold on;
scatter(actual_values, predicted_model2,50,'green', 'Marker', 'diamond', 'DisplayName', sprintf('ProposedModel(R^2 = %.4f)', R_squared_model2));
scatter(actual_values, predicted_model3,50,'red', 'Marker', 'o', 'DisplayName', sprintf('Informer(R^2 = %.4f)', R_squared_model3));
scatter(actual_values, predicted_model4,20,'blue', 'Marker', '*', 'DisplayName', sprintf('CNN-LSTM(R^2 = %.4f)', R_squared_model4));
% 添加轴标签和图标题
xlabel('Y1 True value');
ylabel('Y1 Prediction results');
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

% %% Y2
% actual_values = truth_y2;
% predicted_model1 = predict_y2_ResLSTMNet_GAT;
% predicted_model2 = predict_y2_LSTM;
% predicted_model3 = predict_y2_GRU;
% predicted_model4 = predict_y2_CNN;
% 
% % R平方
% R_squared_model1 = 1 - sum((actual_values - predicted_model1).^2) / sum((actual_values - mean(actual_values)).^2);
% R_squared_model2 = 1 - sum((actual_values - predicted_model2).^2) / sum((actual_values - mean(actual_values)).^2);
% R_squared_model3 = 1 - sum((actual_values - predicted_model3).^2) / sum((actual_values - mean(actual_values)).^2);
% R_squared_model4 = 1 - sum((actual_values - predicted_model4).^2) / sum((actual_values - mean(actual_values)).^2);
% % 散点图
% figure;
% scatter(actual_values, predicted_model1,300,'black', 'Marker','.', 'DisplayName', sprintf('ResLSTMNet-GAT(R^2 = %.4f)', R_squared_model1));
% hold on;
% scatter(actual_values, predicted_model2,80,'red', 'Marker', 'diamond', 'DisplayName', sprintf('LSTM(R^2 = %.4f)', R_squared_model2));
% scatter(actual_values, predicted_model3,80,'green', 'Marker', 'o', 'DisplayName', sprintf('GRU(R^2 = %.4f)', R_squared_model3));
% scatter(actual_values, predicted_model4,50,'blue', 'Marker', '*', 'DisplayName', sprintf('CNN(R^2 = %.4f)', R_squared_model4));
% % 添加轴标签和图标题
% xlabel('Y2 True value');
% ylabel('Y2 Prediction results');
% % title('R-square scatter plot of Five models');
% % 显示图例
% legend('Location', 'southeast');
% 
% % 添加一条45度斜线，用于比较
% % plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'Color', 'k', HandleVisibility='off');
% plot([min(actual_values), 5400], [min(actual_values), 5400], '-', 'Color', 'k', HandleVisibility='off');
% 
% % xlim([min(actual_values), max(actual_values)]);
% % ylim([min(actual_values), max(actual_values)]);
% xlim([min(actual_values), 5300]);
% ylim([min(actual_values), 5300]);
% % 调整图的显示
% % axis equal;
% grid on;