clear
clc

%% Proposed Model
load ..\My_Informer_CO_EN_16r1c_VMD_1_test_result_inversed.mat
mse = [];
mse_y1_1 = mean((truth_y1 - predict_y1_My_Informer).^2);

mse = [mse; mse_y1_1];
%% Proposed Model Without EN&VMD
load ..\My_Informer_CO_data_26r1c_test_result_inversed.mat
mse_y1_2 = mean((truth_y1 - predict_y1_My_Informer).^2);

mse = [mse; mse_y1_2];
%% Informer
load ..\Informer_CO_data_26r1c_test_result_inversed.mat
mse_y1_3 = mean((truth_y1 - predict_y1_Informer).^2);

mse = [mse; mse_y1_3];

%% CNN-LSTM
load ..\cnn_lstm_test_result_inversed.mat
mse_y1_4 = mean((truth_y1 - predict_y1_cnn_lstm).^2);

mse = [mse; mse_y1_4];

mse