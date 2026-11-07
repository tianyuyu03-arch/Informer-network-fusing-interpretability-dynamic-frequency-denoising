prediction = [];
truth = [];
load ..\My_Informer_VMD\weight0\My_Informer_CO_data_EN_16r1c_VMD_0_1_test_result_inversed.mat
prediction = [prediction predict_y1_My_Informer];
truth = [truth truth_y1];
load ..\My_Informer_VMD\weight0\My_Informer_CO_data_EN_16r1c_VMD_0_2_test_result_inversed.mat
prediction = [prediction predict_y1_My_Informer];
truth = [truth truth_y1];
load ..\My_Informer_VMD\weight0\My_Informer_CO_data_EN_16r1c_VMD_0_3_test_result_inversed.mat
prediction = [prediction predict_y1_My_Informer];
truth = [truth truth_y1];
load ..\My_Informer_VMD\weight0\My_Informer_CO_data_EN_16r1c_VMD_0_4_test_result_inversed.mat
prediction = [prediction predict_y1_My_Informer];
truth = [truth truth_y1];
load ..\My_Informer_VMD\weight0\My_Informer_CO_data_EN_16r1c_VMD_0_5_test_result_inversed.mat
prediction = [prediction predict_y1_My_Informer];
truth = [truth truth_y1];
load ..\My_Informer_VMD\weight0\My_Informer_CO_data_EN_16r1c_VMD_0_6_test_result_inversed.mat
prediction = [prediction predict_y1_My_Informer];
truth = [truth truth_y1];
load ..\My_Informer_VMD\weight0\My_Informer_CO_data_EN_16r1c_VMD_0_7_test_result_inversed.mat
prediction = [prediction predict_y1_My_Informer];
truth = [truth truth_y1];
load ..\My_Informer_VMD\weight0\My_Informer_CO_data_EN_16r1c_VMD_0_8_test_result_inversed.mat
prediction = [prediction predict_y1_My_Informer];
truth = [truth truth_y1];
load ..\My_Informer_VMD\weight0\My_Informer_CO_data_EN_16r1c_VMD_0_9_test_result_inversed.mat
prediction = [prediction predict_y1_My_Informer];
truth = [truth truth_y1];
load ..\My_Informer_VMD\weight0\My_Informer_CO_data_EN_16r1c_VMD_0_10_test_result_inversed.mat
prediction = [prediction predict_y1_My_Informer];
truth = [truth truth_y1];
predict_y1_My_Informer = prediction;
truth_y1 = truth;

save ..\My_Informer_CO_EN_16r1c_VMD_0_test_result_inversed.mat truth_y1 predict_y1_My_Informer
