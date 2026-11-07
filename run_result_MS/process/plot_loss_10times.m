close all
clear
clc


load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_1_loss.mat
loss_My_Informer1 = loss_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_2_loss.mat
loss_My_Informer2 = loss_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_3_loss.mat
loss_My_Informer3 = loss_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_4_loss.mat
loss_My_Informer4 = loss_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_5_loss.mat
loss_My_Informer5 = loss_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_6_loss.mat
loss_My_Informer6 = loss_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_7_loss.mat
loss_My_Informer7 = loss_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_8_loss.mat
loss_My_Informer8 = loss_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_9_loss.mat
loss_My_Informer9 = loss_My_Informer;
load ..\My_Informer_VMD\weight1\My_Informer_CO_data_EN_16r1c_VMD_1_10_loss.mat
loss_My_Informer10 = loss_My_Informer;

set(groot, 'DefaultAxesFontName', 'Times New Roman'); % 设置默认字体
figure(1)

% plot(loss_My_Informer1,'k','LineWidth',1.5);
% hold on;
% plot(loss_My_Informer2,'r','LineWidth',1.5);
% hold on;
% plot(loss_My_Informer3,'b','LineWidth',1.5);
% hold on;

plot(loss_My_Informer1,'Color', [1,0,0],'LineWidth',2.5);
hold on;
plot(loss_My_Informer2,'Color', [0,1,0],'LineWidth',2.5);
hold on;
plot(loss_My_Informer3,'Color', [0,0,1],'LineWidth',2.5);
hold on;
plot(loss_My_Informer4,'Color', [1,1,0],'LineWidth',2.5);
hold on;
plot(loss_My_Informer5,'Color', [0,1,1],'LineWidth',2.5);
hold on;
plot(loss_My_Informer6,'Color', [1,0,1],'LineWidth',2.5);
hold on;
plot(loss_My_Informer7,'Color', [1,0.647,0],'LineWidth',2.5);
hold on;
plot(loss_My_Informer8,'Color', [0.502,0,0.502],'LineWidth',2.5);
hold on;
plot(loss_My_Informer9,'Color', [0.647,0.165,0.165],'LineWidth',2.5);
hold on;
plot(loss_My_Informer10,'Color', [0.753,0.753,0.753],'LineWidth',2.5);
hold on;
% plot(loss_My_Informer8,'Color','LineWidth',1.5);

% plot(loss_CNN,':m','LineWidth',1.5);
% hold on;

h=legend('Retraining1','Retraining2','Retraining3','Retraining4','Retraining5','Retraining6','Retraining7','Retraining8','Retraining9','Retraining10');
set(h,'FontSize',8);
grid on
xlabel('Epoch');
ylabel('MSE Loss');
