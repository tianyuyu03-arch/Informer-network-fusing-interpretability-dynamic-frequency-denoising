clear
clc

R2_test = [];
%% Proposed Model
load ..\My_Informer_CO_EN_16r1c_VMD_1_test_result_inversed.mat
actual_values = truth_y1;
predicted_model1 = predict_y1_My_Informer;

r_squared1 = 1 - sum((actual_values - predicted_model1).^2) / sum((actual_values - mean(actual_values)).^2);

% corr_matrix = corrcoef(y_true, y_predict);
% r_squared1 = corr_matrix(1, 2)^2;

R2_test = [R2_test; r_squared1];
%% Proposed Model Without EN&VMD
load ..\My_Informer_CO_data_26r1c_test_result_inversed.mat
actual_values = truth_y1;
predicted_model1 = predict_y1_My_Informer;

r_squared2 = 1 - sum((actual_values - predicted_model1).^2) / sum((actual_values - mean(actual_values)).^2);

R2_test = [R2_test; r_squared2];
%% Informer
load ..\Informer_CO_data_26r1c_test_result_inversed.mat
actual_values = truth_y1;
predicted_model1 = predict_y1_Informer;

r_squared3 = 1 - sum((actual_values - predicted_model1).^2) / sum((actual_values - mean(actual_values)).^2);

R2_test = [R2_test; r_squared3];
%% CNN-LSTM
load ..\cnn_lstm_test_result_inversed.mat
actual_values = truth_y1;
predicted_model1 = predict_y1_cnn_lstm;

r_squared4 = 1 - sum((actual_values - predicted_model1).^2) / sum((actual_values - mean(actual_values)).^2);

R2_test = [R2_test; r_squared4];

R2_test