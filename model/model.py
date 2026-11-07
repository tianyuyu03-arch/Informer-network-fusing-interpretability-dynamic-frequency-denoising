import torch.nn as nn

from .Decoder import Decoder
from .Encoder import Encoder
from .embed import DataEmbedding


class My_Informer(nn.Module):
    def __init__(
        self,
        encinsize=10,
        middlesize=512,
        d_ff=512,
        outsize=3,
        n_heads=8,
        seq_len=96,
        label_len=48,
        pred_len=1,
    ):
        super().__init__()
        self.outsize = outsize
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.bn = nn.BatchNorm1d(encinsize)
        self.lstm = nn.LSTM(encinsize, 32, batch_first=True)
        self.enc_embedding = DataEmbedding(emb_insize=32, emb_outsize=middlesize)
        self.dec_embedding = DataEmbedding(emb_insize=encinsize, emb_outsize=middlesize)
        self.encoder = Encoder(middlesize, d_ff, n_heads)
        self.fnn1 = nn.Linear(middlesize, middlesize * 2, bias=True)
        self.fnn2 = nn.Linear(middlesize * 2, middlesize, bias=True)
        self.decoder = Decoder(middlesize, d_ff, n_heads)
        self.projection = nn.Linear(middlesize, outsize, bias=True)

    def forward(
        self,
        x_enc,
        x_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        x_enc = self.bn(x_enc.transpose(1, 2)).transpose(1, 2)
        x_dec = self.bn(x_dec.transpose(1, 2)).transpose(1, 2)
        x_enc, _ = self.lstm(x_enc)

        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, enc_self_mask)
        enc_out = self.fnn2(self.fnn1(enc_out))

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
        )
        dec_out = self.projection(dec_out)
        return dec_out[:, -self.pred_len :, :]
