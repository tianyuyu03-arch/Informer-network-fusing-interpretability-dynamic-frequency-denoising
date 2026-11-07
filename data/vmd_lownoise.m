function [input_train_new] = vmd_lownoise(input_train,high_weight)

K = 2;
high_fqs = [];
low_fqs = [];
for i = 1:size(input_train, 2)-1
    real_u=input_train(:,i);
    [imf,residual] = vmd(real_u,'NumIMF', K,'PenaltyFactor',100);
    high_fq = imf(:, 1);
    low_fq = imf(:, 2);
    high_fqs = [high_fqs, high_fq];
    low_fqs = [low_fqs, low_fq];
end
cleanX = low_fqs+high_weight*high_fqs ;
input_train_new = [cleanX input_train(:, end)];

end