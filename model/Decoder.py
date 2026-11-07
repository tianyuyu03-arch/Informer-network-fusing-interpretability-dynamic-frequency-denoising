import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attn import AttentionLayer, ProbAttention, FullAttention

class Decoder(nn.Module):
    def __init__(self, middlesize, d_ff, n_heads):
        super(Decoder,self).__init__()
        self.dec_layer = DecoderLayer(middlesize, d_ff, n_heads)
        self.norm_layer = nn.LayerNorm(middlesize)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
    # 即dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        x = self.dec_layer(x, cross, x_mask=x_mask, cross_mask=cross_mask) # [256, 72, 512]

        return self.norm_layer(x)

class DecoderLayer(nn.Module):
    def __init__(self, middlesize, d_ff, n_heads):
        super(DecoderLayer,self).__init__()
        # self.lstm1 = nn.LSTM(middlesize, 32, batch_first=True)
        # self.lstm2 = nn.LSTM(32, 64, batch_first=True)
        # self.lstm3 = nn.LSTM(64, middlesize, batch_first=True)

        self.self_attn = AttentionLayer(ProbAttention(True, factor=5, output_attention=False),
                                          middlesize, n_heads)
        self.cross_attn = AttentionLayer(FullAttention(False, factor=5, output_attention=False),
                                          middlesize, n_heads)
        self.conv1 = nn.Conv1d(in_channels=middlesize, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=middlesize, kernel_size=1)
        self.norm1 = nn.LayerNorm(middlesize)
        self.norm2 = nn.LayerNorm(middlesize)
        self.norm3 = nn.LayerNorm(middlesize)
        self.activation = F.relu
    def forward(self, x, cross, x_mask=None, cross_mask=None):

        x_attn = self.self_attn(x, x, x, attn_mask=x_mask)[0]

        x = x + x_attn

        x = self.norm1(x) # [128, 6, 32]

        x_attn = self.cross_attn(x, cross, cross, attn_mask=cross_mask)[0] # [256, 72, 512]
        x = x + x_attn
        y = x = self.norm2(x) # [256, 72, 512]

        y = self.activation(self.conv1(y.transpose(-1, 1))) # [256, 512, 72]
        y = self.conv2(y).transpose(-1, 1) # [256, 72, 512]

        return self.norm3(x + y)