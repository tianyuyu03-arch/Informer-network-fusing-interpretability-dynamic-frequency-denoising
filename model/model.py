import torch.nn as nn
from .embed import DataEmbedding
# from .model_Encoder import Encoder
from .Encoder import Encoder
from .Decoder import Decoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class My_Informer(nn.Module):
    def __init__(self, encinsize=10, middlesize=512, d_ff=512, outsize=3, n_heads=8, seq_len=96, label_len=48, pred_len=1):
        super(My_Informer,self).__init__()
        self.outsize = outsize
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.bn = nn.BatchNorm1d(encinsize)
        self.lstm = nn.LSTM(encinsize, 32)

        self.enc_embedding = DataEmbedding(emb_insize=32, emb_outsize=middlesize)
        self.dec_embedding = DataEmbedding(emb_insize=encinsize, emb_outsize=middlesize)
        # x_enc和x_dec只有序列长度不同 其他维度相同

        self.encoder = Encoder(middlesize, d_ff, n_heads)

        self.fnn1 = nn.Linear(middlesize, middlesize * 2, bias=True)
        self.fnn2 = nn.Linear(middlesize * 2, middlesize, bias=True)

        self.decoder = Decoder(middlesize, d_ff, n_heads)

        self.projection = nn.Linear(middlesize, outsize, bias=True)

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None): # [256, 96, 10] [256, 72, 10]
        x_enc = self.bn(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        x_dec = self.bn(x_dec.permute(0, 2, 1)).permute(0, 2, 1)

        x_enc, _ = self.lstm(x_enc)

        enc_out = self.enc_embedding(x_enc)  # [64, 15, 27] [128, 48, 64]
        enc_out, attns = self.encoder(enc_out, enc_self_mask) #   # 元组：包含输出以及attns [128, 12, 64]
        # 编码完毕

        enc_out = self.fnn1(enc_out)
        enc_out = self.fnn2(enc_out)
        # FNN层

        dec_out = self.dec_embedding(x_dec)  # [256, 72, 512]
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask) # [256, 72, 512]
        # 解码完毕
        dec_out = self.projection(dec_out) # [256, 72, 3]
        # dec_out:[256, 72, 12] 但是72个其中 48个是有标签的 带我们的 要的是后面24个
        pred = dec_out[:, -self.pred_len:, :]
        return pred  # 选后面24个 为预测值