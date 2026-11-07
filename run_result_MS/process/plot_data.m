data = readmatrix("CO_26r1c.csv");
x = data(:, 1:26);
y = data(:, end);

figure
for i=1:26
    subplot(7, 4, i);
    plot(x(:, i), 'r','LineWidth',0.25)
    iStr = num2str(i);    % 将数字转换为字符
    label_y = ['Variabel X', iStr]; 
    ylabel(label_y)

end