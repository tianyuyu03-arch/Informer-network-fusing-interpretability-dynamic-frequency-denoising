import torch
import torch.nn as nn
from .attn import ProbAttention, AttentionLayer
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, middlesize, d_ff, n_heads):
        super(Encoder, self).__init__()

        self.enc_layer = EncoderLayer(middlesize=middlesize, d_ff=d_ff, n_heads=n_heads)
        self.conv_layer = ConvLayer(middlesize=middlesize)
        self.attn_layer = AttentionLayer(ProbAttention(False, factor=5, attention_dropout=0.0, output_attention=False),
                                         middlesize, n_heads)
        self.norm_layer = torch.nn.LayerNorm(middlesize)

    def forward(self, x, attn_mask=None):
        attns = []
        for i in range(6):
            x, attn = self.enc_layer(x, attn_mask) # [256, 96, 512], None
            attns.append(attn)
            x = self.conv_layer(x) # [256, 48, 512] # 做一个卷积,后面还有pooling，长度减半,变为48,相当于一个采样的操作，"蒸馏"
        x, attn = self.attn_layer(x, x, x, attn_mask)  # 按照原理，再对长48的x做attention，输出维度不变[256, 48, 512]
        attns.append(attn)
        x = self.norm_layer(x)
        return x, attns

class EncoderLayer(nn.Module):
    def __init__(self, middlesize, d_ff, n_heads):
        super(EncoderLayer,self).__init__()
        self.attn_layer = AttentionLayer(ProbAttention(False, factor=5, attention_dropout=0.0, output_attention=False),
                                         middlesize, n_heads)
        self.conv1 = nn.Conv1d(in_channels=middlesize, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=middlesize, kernel_size=1)
        self.norm1 = nn.LayerNorm(middlesize)
        self.norm2 = nn.LayerNorm(middlesize)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attn_layer(x, x, x, attn_mask)  # [256, 96, 512] None

        x = x + self.dropout(new_x)  # attention后经过fc，再加一个残差连接

        # 下面三步不清楚 对y卷积两次之后再和x相加
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class ConvLayer(nn.Module):
    def __init__(self, middlesize):
        super(ConvLayer,self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=middlesize,
                                  out_channels=middlesize,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(middlesize)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    def forward(self, x): # [256, 96, 512]
        x = self.downConv(x.permute(0, 2, 1)) # [256, 512, 96]
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x) # [256, 512, 48]
        x = x.transpose(1, 2) # [256, 48, 512]
        return x