clear
clc
close all
% num返回的是excel中的数据，txt输出的是文本内容，raw输出的是未处理数据
x = csvread('CO_EN_16r1c.csv') ;

x1 = x(2:end,1);
x2 = x(2:end,2);
x3 = x(2:end,3);
x4 = x(2:end,4);
x5 = x(2:end,5);
x6 = x(2:end,6);
x7 = x(2:end,7);
x8 = x(2:end,8);
x9 = x(2:end,9);
x10 = x(2:end,10);
x11 = x(2:end,11);
x12 = x(2:end,12);
x13 = x(2:end,13);
x14 = x(2:end,14);
x15 = x(2:end,15);
x16 = x(2:end,16);
% x17 = x(:,17);
% x18 = x(:,18);
% x19 = x(:,19);
% x20 = x(:,20);
% x21 = x(:,21);
% x22 = x(:,22);
% x23 = x(:,23);
% x24 = x(:,24);
% x25 = x(:,25);
% x26 = x(:,26);
y = x(2:end,end);

figure
% subplot(4,1,1);
plot(x1);
% subtitle('x1');
figure
% subplot(4,1,2);
plot(x2);
% subtitle('x2');
figure
% subplot(4,1,3);
plot(x3);
% subtitle('x3');
xticks([]);
xlabel('');
yticks([]);
ylabel('');

figure
% subplot(4,1,4);
plot(x4);
% subtitle('x4');
xticks([]);
xlabel('');
yticks([]);
ylabel('');

figure
% subplot(4,1,1);
plot(x5);
% subtitle('x5');
xticks([]);
xlabel('');
yticks([]);
ylabel('');

figure
% subplot(4,1,2);
plot(x6);
% subtitle('x6');
xticks([]);
xlabel('');
yticks([]);
ylabel('');

figure
% subplot(4,1,3);
plot(x7);
% subtitle('x7');
xticks([]);
xlabel('');
yticks([]);
ylabel('');

figure
% subplot(4,1,4);
plot(x8);
% subtitle('x8');
xticks([]);
xlabel('');
yticks([]);
ylabel('');


figure
subplot(4,1,1);
plot(x9);
subtitle('x9');
subplot(4,1,2);
plot(x10);
subtitle('x10');
subplot(4,1,3);
plot(x11);
subtitle('x11');
subplot(4,1,4);
plot(x12);
subtitle('x12');

figure
subplot(4,1,1);
plot(x13);
subtitle('x13');
subplot(4,1,2);
plot(x14);
subtitle('x14');
subplot(4,1,3);
plot(x15);
subtitle('x15');

figure
plot(x16);
xticks([]);
xlabel('');
yticks([]);
ylabel('');



figure
plot(y, 'Color', 'r');
xticks([]);
xlabel('');
yticks([]);
ylabel('');