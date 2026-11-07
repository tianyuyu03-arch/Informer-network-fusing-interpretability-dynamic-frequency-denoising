load ..\My_Informer_CO_EN_16r1c_VMD_0_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared0 = corr_matrix(1, 2)^2;
[r_squared0 rmse] = rsquare(y_true, y_predict);
load ..\My_Informer_CO_EN_16r1c_VMD_1_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared1 = corr_matrix(1, 2)^2;
[r_squared1 rmse] = rsquare(y_true, y_predict);

load ..\My_Informer_CO_EN_16r1c_VMD_2_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared2 = corr_matrix(1, 2)^2;
[r_squared2 rmse] = rsquare(y_true, y_predict);

load ..\My_Informer_CO_EN_16r1c_VMD_3_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared3 = corr_matrix(1, 2)^2;
[r_squared3 rmse] = rsquare(y_true, y_predict);

load ..\My_Informer_CO_EN_16r1c_VMD_4_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared4 = corr_matrix(1, 2)^2;
[r_squared4 rmse] = rsquare(y_true, y_predict);

load ..\My_Informer_CO_EN_16r1c_VMD_5_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared5 = corr_matrix(1, 2)^2;
[r_squared5 rmse] = rsquare(y_true, y_predict);

load ..\My_Informer_CO_EN_16r1c_VMD_6_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared6 = corr_matrix(1, 2)^2;
[r_squared6 rmse] = rsquare(y_true, y_predict);

load ..\My_Informer_CO_EN_16r1c_VMD_7_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared7 = corr_matrix(1, 2)^2;
[r_squared7 rmse] = rsquare(y_true, y_predict);

load ..\My_Informer_CO_EN_16r1c_VMD_8_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared8 = corr_matrix(1, 2)^2;
[r_squared8 rmse] = rsquare(y_true, y_predict);

load ..\My_Informer_CO_EN_16r1c_VMD_9_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared9 = corr_matrix(1, 2)^2;
[r_squared9 rmse] = rsquare(y_true, y_predict);

load ..\My_Informer_CO_EN_16r1c_VMD_10_test_result_inversed.mat

y_true = truth_y1;
y_predict = predict_y1_My_Informer;

corr_matrix = corrcoef(y_true, y_predict);
r_squared10 = corr_matrix(1, 2)^2;
[r_squared10 rmse] = rsquare(y_true, y_predict);

r_squared(1,1) = r_squared0;
r_squared(1,2) = r_squared1;
r_squared(1,3) = r_squared2;
r_squared(1,4) = r_squared3;
r_squared(1,5) = r_squared4;
r_squared(1,6) = r_squared5;
r_squared(1,7) = r_squared6;
r_squared(1,8) = r_squared7;
r_squared(1,9) = r_squared8;
r_squared(1,10) = r_squared9;
r_squared(1,11) = r_squared10;

%% 画图
data = r_squared;
% 绘制柱形图
figure;
bar(data, 'FaceColor', 'r');

% 设置 y 轴的显示范围为 0.8 到 1.0
ylim([0.7, 0.95]);

% 显示标签
for i = 1:length(data)
    text(i, data(i), num2str(data(i), '%.3f'), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% 更改横坐标的刻度值
xticks(1:length(data));  % 设置刻度值的位置
xticklabels({'0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'});  % 设置刻度值的标签

% 添加标题和标签
% title('柱形图示例');
xlabel('The weight of the high-frequency modes');
ylabel('R^{2}');

% 连接各柱体的顶点
hold on;
plot(1:length(data), data, 'o--', 'Color', 'k', 'MarkerFaceColor', 'k' ,'LineWidth', 1.5, 'MarkerSize', 3);
hold off;