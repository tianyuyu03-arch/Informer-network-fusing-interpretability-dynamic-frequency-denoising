import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module): # 裁剪的模块，裁剪多出来的padding
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class DCConv(nn.Module):  # 用这个的话Conv1d中的padding要设置为原来的2倍！
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(DCConv,self).__init__()
        self.conv= weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding)

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.conv(x)
        out = self.chomp(x)
        return out

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TCNBlock,self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.dc_conv1 = DCConv(in_channels, out_channels, 3, 1, 2, 1)
        # self.bn2 = nn.BatchNorm1d(in_channels * 2)
        # self.dc_conv2 = DCConv(in_channels * 2, out_channels, 3, 1, 4, 2)
        self.rl = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # y = x
        x = self.bn1(x)
        x = self.dc_conv1(x)
        x = self.rl(x)
        x = self.dropout(x)  # 删掉效果变差
        return x
        # return x + y


if __name__ == '__main__':
    # input_data = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=torch.float32)
    input_data = torch.randn((3, 6, 4)) # [batch_size, seq_len, channels]
    input_data = input_data.permute(0, 2, 1)
    # [3, 4, 6]

    model = TCNBlock(input_data.size()[1], 24)
    out = model(input_data)

    # conv1d = nn.Conv1d(4, 8, 3, 1, 1)
    # conv_out_data = conv1d(input_data)
    # dcconv = DCConv(4, 8, 3, 1, 2)
    # dcconv_out_data = dcconv(input_data)
    print('结束')