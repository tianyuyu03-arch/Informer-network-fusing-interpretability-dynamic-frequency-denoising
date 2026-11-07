import torch
import torch.nn as nn
import math
class TokenEmbedding(nn.Module):
    def __init__(self, emb_insize, emb_outsize):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(emb_insize, emb_outsize, 3, 1, 1)
        self.tokenLstm = nn.LSTM(emb_insize, emb_outsize, batch_first=True)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):  # x:[256, 96, 10]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)  # [256, 96, 32]
        # x, _ = self.tokenLstm(x)
        # x = tuple(self.tokenLstm(x)[0])  # [256, 96, 32]
        return x
class PositionalEmbedding(nn.Module):
    def __init__(self, emb_outsize, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, emb_outsize).float()  # [5000, 64]
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # [5000, 1]：0 1 2 3 4 ...
        div_term = (torch.arange(0, emb_outsize, 2).float() * -(math.log(10000.0) / emb_outsize)).exp()  # 32

        pe[:, 0::2] = torch.sin(position * div_term) # 从0开始 每隔两个选一个元素 即取偶数列
        pe[:, 1::2] = torch.cos(position * div_term) # 取奇数列

        pe = pe.unsqueeze(0)  # pe:[1, 5000, 64]
        self.register_buffer('pe', pe)

    def forward(self, x):  # x:[128, 256, 32]
        return self.pe[:, :x.size(1)]  # [128, 256, 64]

class DataEmbedding(nn.Module):
    def __init__(self, emb_insize = 10, emb_outsize = 32):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(emb_insize=emb_insize, emb_outsize=emb_outsize) # TokenEmbedding就是一维卷积，将维度由c_in变为d_model
        self.position_embedding = PositionalEmbedding(emb_outsize=emb_outsize)
        self.dropout = nn.Dropout(0)
    def forward(self, x):  # [128, 256, 32]
        x = self.value_embedding(x) + self.position_embedding(x)  # [256, 96, 64]
        # 全部映射到d_model维再相加
        return self.dropout(x)  # [256, 96, 32]