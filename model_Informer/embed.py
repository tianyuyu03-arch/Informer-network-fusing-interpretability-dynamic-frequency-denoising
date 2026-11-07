import torch
import torch.nn as nn
import math
class TokenEmbedding(nn.Module):
    def __init__(self, encinsize, middlesize):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(encinsize, middlesize, 3, 1, 1);

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):  # x:[256, 96, 10]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)  # [256, 96, 32]
        return x
class PositionalEmbedding(nn.Module):
    def __init__(self, middlesize, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, middlesize).float()  # [5000, 512]
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, middlesize, 2).float() * -(math.log(10000.0) / middlesize)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # pe:[1, 5000, 512],x:[256, 96, 32]
        return self.pe[:, :x.size(1)]  # [1, 96, 512]

class DataEmbedding(nn.Module):
    def __init__(self, encinsize = 10, middlesize = 32):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(encinsize=encinsize, middlesize=middlesize) # TokenEmbedding就是一维卷积，将维度由c_in变为d_model
        self.position_embedding = PositionalEmbedding(middlesize=middlesize)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):  # [256, 96, 10]
        x = self.value_embedding(x) + self.position_embedding(x)  # [256, 96, 32]
        # 全部映射到d_model维再相加
        return self.dropout(x)  # [256, 96, 32]