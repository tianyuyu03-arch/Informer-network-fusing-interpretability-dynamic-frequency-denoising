import torch
import torch.nn as nn
import numpy as np
class Multi_Stream_LSTM(nn.Module):
    def __init__(self, middelsize):
        super(Multi_Stream_LSTM, self).__init__()
        self.lstm = nn.LSTM(middelsize, middelsize, batch_first=True)

    def forward(self, x):
        batch_size, length, channels = x.size()
        branch1 = x[:, 0:length // 3, :]
        branch2 = x[:, length // 3:length - length // 3, :]
        branch3 = x[:, length - length // 3:length, :]
        branch1, _ = self.lstm(branch1)
        branch2, _ = self.lstm(branch2)
        branch3, _ = self.lstm(branch3)
        out = torch.cat([branch1, branch2, branch3], axis=1)
        return out

if __name__ == '__main__':
    A = torch.randn((32, 10, 16))
    mslstm = Multi_Stream_LSTM(16)
    B = mslstm(A)
    print('结束')