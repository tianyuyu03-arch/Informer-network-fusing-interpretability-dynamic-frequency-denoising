clear
clc

load ../test_result_inverse.mat

mae = [];
%% ResLSTMNet_GAT
mae_y1_1 = mean(abs(truth_y1 - predict_y1_ResLSTMNet_GAT));
mae_y2_1 = mean(abs(truth_y2 - predict_y2_ResLSTMNet_GAT));

mae = [mae; mae_y1_1 mae_y2_1];
%% LSTM
mae_y1_2 = mean(abs(truth_y1 - predict_y1_LSTM));
mae_y2_2 = mean(abs(truth_y2 - predict_y2_LSTM));

mae = [mae; mae_y1_2 mae_y2_2];
%% GRU
mae_y1_3 = mean(abs(truth_y1 - predict_y1_GRU));
mae_y2_3 = mean(abs(truth_y2 - predict_y2_GRU));

mae = [mae; mae_y1_3 mae_y2_3];
%% CNN
mae_y1_4 = mean(abs(truth_y1 - predict_y1_CNN));
mae_y2_4 = mean(abs(truth_y2 - predict_y2_CNN));

mae = [mae; mae_y1_4 mae_y2_4];

mae