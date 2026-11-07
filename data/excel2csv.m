% num返回的是excel中的数据，txt输出的是文本内容，raw输出的是未处理数据
[data,txt,raw]=xlsread('CO_data_26r1c.xlsx') ;

file_name = 'CO_data_26r1c.csv';

writematrix(data,file_name);