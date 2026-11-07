close all
clear
clc
%% EN-VMD-ProposedInformer拟合效果作图
prediction = [];
truth = [];
load ..\My_Informer_CO_EN_16r1c_VMD_3_test_result_inversed.mat
figure(1)
subplot(4,1,1)
hold on;
plot(truth_y1,'k','LineWidth',1.5);
hold on
plot(predict_y1_My_Informer,'--b','LineWidth',1.5);
hold on;
h=legend('Truth','Proposed Model');
set(h,'FontSize',8);
grid on
xlabel('Time');
ylabel('Y1 output');
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
