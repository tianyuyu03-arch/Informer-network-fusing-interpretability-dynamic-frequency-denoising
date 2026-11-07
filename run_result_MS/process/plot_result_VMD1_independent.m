close all
clear
clc
%% EN-VMD-ProposedInformer拟合效果作图
prediction = [];
truth = [];
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_1_test_result_inversed.mat
truth_y1_1 = truth_y1;
predict_y1_My_Informer1 = predict_y1_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_2_test_result_inversed.mat
truth_y1_2 = truth_y1;
predict_y1_My_Informer2 = predict_y1_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_3_test_result_inversed.mat
truth_y1_3 = truth_y1;
predict_y1_My_Informer3 = predict_y1_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_4_test_result_inversed.mat
truth_y1_4 = truth_y1;
predict_y1_My_Informer4 = predict_y1_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_5_test_result_inversed.mat
truth_y1_5 = truth_y1;
predict_y1_My_Informer5 = predict_y1_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_6_test_result_inversed.mat
truth_y1_6 = truth_y1;
predict_y1_My_Informer6 = predict_y1_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_7_test_result_inversed.mat
truth_y1_7 = truth_y1;
predict_y1_My_Informer7 = predict_y1_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_8_test_result_inversed.mat
truth_y1_8 = truth_y1;
predict_y1_My_Informer8 = predict_y1_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_9_test_result_inversed.mat
truth_y1_9 = truth_y1;
predict_y1_My_Informer9 = predict_y1_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_10_test_result_inversed.mat
truth_y1_10 = truth_y1;
predict_y1_My_Informer10 = predict_y1_My_Informer;

figure(1)
subplot(2,5,1)
hold on;
plot(truth_y1_1,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer1,'--b','LineWidth',1.5);
hold on;
h=legend('Truth','Proposed Model');
set(h,'FontSize',8);
grid on
xlabel('Time');
ylabel('Y1 output');

subplot(2,5,2)
hold on;
plot(truth_y1_2,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer2,'--b','LineWidth',1.5);
hold on;
grid on
% xlabel('Time');
% ylabel('Y1 output');

subplot(2,5,3)
hold on;
plot(truth_y1_3,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer3,'--b','LineWidth',1.5);
hold on;
grid on

subplot(2,5,4)
hold on;
plot(truth_y1_4,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer4,'--b','LineWidth',1.5);
hold on;
grid on

subplot(2,5,5)
hold on;
plot(truth_y1_5,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer5,'--b','LineWidth',1.5);
hold on;
grid on

subplot(2,5,6)
hold on;
plot(truth_y1_6,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer6,'--b','LineWidth',1.5);
hold on;
grid on

subplot(2,5,7)
hold on;
plot(truth_y1_7,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer7,'--b','LineWidth',1.5);
hold on;
grid on

subplot(2,5,8)
hold on;
plot(truth_y1_8,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer8,'--b','LineWidth',1.5);
hold on;
grid on

subplot(2,5,9)
hold on;
plot(truth_y1_9,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer9,'--b','LineWidth',1.5);
hold on;
grid on

subplot(2,5,10)
hold on;
plot(truth_y1_10,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer10,'--b','LineWidth',1.5);
hold on;
grid on

% axis([0 1000 0.24 0.30])
%% ProposedInformer拟合效果作图
load ..\My_Informer_CO_data_26r1c_test_result_inversed.mat
subplot(4,1,2)
hold on;
plot(truth_y1,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer,'--b','LineWidth',1.5);
hold on;
h=legend('Truth','Proposed Model without VMD&EN');
set(h,'FontSize',8);
grid on
xlabel('Time');
ylabel('Y1 output');
xlim([0 1000]);
%% Informer拟合效果作图
% figure(2)
load ..\Informer_CO_data_26r1c_test_result_inversed.mat
subplot(4,1,3)
hold on;
plot(truth_y1,'k','LineWidth',1.5);
hold on
plot(predict_y1_Informer,'--b','LineWidth',1.5);
hold on;
h=legend('Truth','Informer');
set(h,'FontSize',8);
grid on
xlabel('Time');
ylabel('Y1 output');
% axis([0 1000 0.24 0.32])
xlim([0 1000]);

%% CNN-LSTM拟合效果作图
load ..\cnn_lstm_test_result_inversed.mat
subplot(4,1,4)
hold on;
plot(truth_y1,'k','LineWidth',1.5);
hold on
plot(predict_y1_cnn_lstm,'--b','LineWidth',1.5);
hold on;
h=legend('Truth','CNN-LSTM');
set(h,'FontSize',8);
grid on
xlabel('Time');
ylabel('Y1 output');
% axis([0 1000 0.24 0.32])
