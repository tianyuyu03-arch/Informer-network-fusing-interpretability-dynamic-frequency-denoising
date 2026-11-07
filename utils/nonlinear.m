clc
clear
close all

% L = 3000;
% Na = 1;
% Nb = 3;
% 
% a1 = 1;
% a2 = 0.5;
% a3 = 0.3;
% a4 = 0.1;
% b1 = -0.9;
% b2 = -0.15;
% b3 = -0.02;
% c1 = 0.7;
% c2 = 1.5;
% 
% std = 0.1;
% 
% w = rand(2,L);
% 
% for k=1:L
%     u(k) = 0.1*sin(0.5*k) + 0.2*cos(0.08*k) + 0.3*sin(0.16*k) ;
%     real_u(k) = u(k)+0.2*w(1,k);
% end
% 
% figure(1)
% plot(u);
% title('输入');
% 
% x=zeros(1,L);
% y=zeros(1,L);
% 
% for k=1+Na:L
%     x(k) = a1*sin(u(k)) + a2*(sin(u(k)).^2) + a3*(sin(u(k)).^3) + a4*(sin(u(k)).^4);
% end
% 
% figure(2)
% plot(x);
% title('非线性层输出');
% 
% 
% for k=1+Nb:L
%     y(k) = b1*y(k-1) + b2*y(k-2) + b3*y(k-3) + c1*x(k) + c2*x(k-1)+std*w(2,k);
% end

% 导入数据 signal，设置参数
K = 3;  % 模态数
alpha = 2000;  % 正则化参数
xx=xlsread('Data_16r6c_outqiyi_foresee.xlsx');
%xx=xlsread('Data_12r6c_5000.xlsx');
real_u=xx(:,30);
% 运行 VMD
[imf,residual] = vmd(real_u,'NumIMF',9,'PenaltyFactor',100);

t1 = tiledlayout(3,3,TileSpacing="compact",Padding="compact");
for n = 1:9
    ax(n) = nexttile(t1);
    plot(imf(:,n)')
    xlim([1 1000])
    txt = ["IMF",num2str(n)];
    title(txt)
    xlabel("Time (s)")
end
title(t1,"Variational Mode Decomposition")

cleanX = sum(imf(:,7:9),2);
figure(5)
plot(real_u);
hold on;
plot(cleanX')
legend("Original X","Clean X")
xlabel("Time (s)")
ylabel("Signal")

data = [real_u cleanX];
save nonlinearData.mat data
error=real_u-cleanX;
mean_value = mean(error);
variance = var(error);

disp("均值：" + mean_value);
disp("方差：" + variance);

columnNames = {'output',  'clear_output'};
dataTable = array2table(data, 'VariableNames', columnNames);

% 指定保存路径和文件名
outputFilename = 'nonlinearData.xlsx';

% 使用writetable函数将数据保存为Excel文件
writetable(dataTable, outputFilename, 'Sheet', 'Sheet1', 'WriteRowNames', true);

