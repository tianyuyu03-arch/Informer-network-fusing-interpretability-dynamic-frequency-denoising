import torch
import torch.nn as nn
from .attn import ProbAttention, AttentionLayer
import torch.nn.functional as F
from utils.multi_stream_lstm import Multi_Stream_LSTM
class Encoder(nn.Module): # 序列长度减半 其他不变
    def __init__(self, middlesize, d_ff, n_heads):
        super(Encoder, self).__init__()
        self.bn = nn.BatchNorm1d(middlesize)

        self.enc_layer = EncoderLayer(middlesize=middlesize, d_ff=d_ff, n_heads=n_heads)
        # EncoderLayer维度不变
        self.conv_layer = ConvLayer(middlesize=middlesize)
        # ConvLayer后序列长度减半 其他不变
        self.attn_layer = AttentionLayer(ProbAttention(False, factor=5, output_attention=False),
                                         middlesize, n_heads)
        self.norm_layer = nn.LayerNorm(middlesize)

    def forward(self, x, attn_mask=None):
        # x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1) 加上这句效果差！
        attns = []
        for i in range(3):
            x, attn = self.enc_layer(x, attn_mask)  # [256, 96, middlesize], None
            attns.append(attn)
            x = self.conv_layer(x)  # [256, 48, 512] # 做一个卷积,后面还有pooling，长度减半,变为48,相当于一个采样的操作，"蒸馏"

        # x = self.conv_layer(x) # [256, 48, 512] # 做一个卷积,后面还有pooling，长度减半,变为48,相当于一个采样的操作，"蒸馏"
        x, attn = self.attn_layer(x, x, x, attn_mask)  # 按照原理，再对长48的x做attention，输出维度不变[256, 48, 512]
        attns.append(attn)
        x = self.norm_layer(x)
        return x, attns

class EncoderLayer(nn.Module):  # 维度不变！
    def __init__(self, middlesize, d_ff, n_heads):
        super(EncoderLayer,self).__init__()

        self.attn_layer = AttentionLayer(ProbAttention(False, factor=5, output_attention=False),
                                         middlesize, n_heads)
        # self.conv1 = nn.Conv1d(in_channels=middlesize, out_channels=d_ff, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=middlesize, kernel_size=3, stride=1, padding=1)
        self.MS_LSTM = Multi_Stream_LSTM(middlesize)
        # self.lstm1 = nn.LSTM(middlesize, 32, batch_first=True)
        # # self.lstm2 = nn.LSTM(32, 64, batch_first=True)
        # self.lstm2 = nn.LSTM(32, middlesize, batch_first=True)
        self.norm1 = nn.LayerNorm(middlesize) # LayerNorm要和第三个维度相等 BatchNorm要和第二个维度相等
        self.norm2 = nn.LayerNorm(middlesize)
        self.activation = F.relu

    def forward(self, x, attn_mask=None):  # [256, 20, 64]

        new_x, attn = self.attn_layer(x, x, x, attn_mask)  # [256, 20, 64] None

        x = x + new_x  # [256, 20, 64] attention后经过fc，再加一个残差连接

        # 下面三步不清楚 对y卷积两次之后再和x相加
        y = self.norm1(x)  # [128， 15， 32]

        y = self.MS_LSTM(y)
        y = self.activation(y)

        out = self.norm2(x + y)
        # out = self.norm2(y)
        return out, attn

class ConvLayer(nn.Module):  # 序列长度减半
    def __init__(self, middlesize):
        super(ConvLayer,self).__init__()
        self.downConv = nn.Conv1d(in_channels=middlesize,
                                  out_channels=middlesize,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(middlesize)
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    def forward(self, x): # [128, 15, 32]
        x = self.downConv(x.permute(0, 2, 1)) # [128, 32, 15] 转换维度 且不变
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x) # [128, 32, 8]
        x = x.transpose(1, 2) # [128, 8, 32]
        return x