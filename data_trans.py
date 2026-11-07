import numpy as np
from sklearn.preprocessing import StandardScaler


def data_trans(data, target_scaler, inpsize, outsize, fit_end):
    """Scale features and target using only rows available to model training."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError('Expected a two-dimensional table.')
    if values.shape[1] < inpsize + outsize:
        raise ValueError(f'Expected at least {inpsize + outsize} numeric columns.')
    if not 0 < fit_end <= len(values):
        raise ValueError('fit_end must select a non-empty training prefix.')
    if not np.isfinite(values).all():
        raise ValueError('Dataset contains missing or non-finite values.')

    inputs = values[:, :inpsize]
    targets = values[:, -outsize:]
    input_scaler = StandardScaler().fit(inputs[:fit_end])
    target_scaler.fit(targets[:fit_end])
    scaled = np.concatenate((input_scaler.transform(inputs), target_scaler.transform(targets)), axis=1)
    return scaled, target_scaler, input_scaler
