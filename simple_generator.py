from resgan_model import ResBlock1D, ResUpSample1D
import torch.nn as nn
import torch


class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.in_conv = nn.Conv1d(in_channels=1,
                                 out_channels=16,
                                 kernel_size=32,
                                 stride=2,
                                 padding=15)
        self.in_bn = nn.BatchNorm1d(16)
        self.in_prelu = nn.PReLU()

        self.down_block = ResBlock1D(in_channel=16, out_channel=64)  # [B x 64 x 4096][B x 64 x 2048]

        self.up= nn.ConvTranspose1d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=32,
                                    stride=2,
                                    padding=15)
        self.bn = nn.BatchNorm1d(64)
        self.prelu = nn.PReLU()

        self.up_block = ResUpSample1D(in_channel=128, out_channel=16)

        self.out_transconv = nn.ConvTranspose1d(in_channels=16,
                                                out_channels=1,
                                                kernel_size=32,
                                                stride=2,
                                                padding=15)
        self.tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_bn(x)
        e0 = self.in_prelu(x)  # e0 16 x 8192

        e1, e2 = self.down_block(e0)  # e1 64 x 4096 e2 64 x 2048

        d1 = self.up(e2)
        d1 = self.bn(d1)
        d1 = self.prelu(d1)  # d1 64 x 4096

        d2 = self.up_block(d1, e1)  # d2 16 x 8192

        out = self.out_transconv(d2)
        out = self.tanh(out)
        return out

