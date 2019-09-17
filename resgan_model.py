import torch.nn as nn
import torch


class ResBlock1D(nn.Module):
    """
    Residual block for down-sample
    """
    def __init__(self, in_channel, out_channel):
        super(ResBlock1D, self).__init__()
        self.downsample = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=4)
        self.conv1 = nn.Conv1d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=32,
                               stride=2,
                               padding=15)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv1d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=32,
                               stride=2,
                               padding=15)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out1 = out
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out2 = self.prelu2(out)
        return out1, out2


class ResUpSample1D(nn.Module):
    """
    Residual block for up sample
    """
    def __init__(self, in_channel, out_channel):
        super(ResUpSample1D, self).__init__()
        self.transconv = nn.ConvTranspose1d(in_channels=in_channel,
                                            out_channels=out_channel,
                                            kernel_size=32,
                                            stride=2,
                                            padding=15)
        self.bn = nn.BatchNorm1d(out_channel)
        self.prelu = nn.PReLU()

    def forward(self, x, skip):
        out = torch.cat((x, skip), 1)
        out = self.transconv(out)
        out = self.bn(out)
        out = self.prelu(out)
        return out


class ResGenerator(nn.Module):
    def __init__(self):
        super(ResGenerator, self).__init__()
        # encoder
        self.in_conv = nn.Conv1d(in_channels=1,
                                 out_channels=16,
                                 kernel_size=32,
                                 stride=2,
                                 padding=15)
        self.in_bn = nn.BatchNorm1d(16)
        self.in_prelu = nn.PReLU()

        self.resblock1 = ResBlock1D(in_channel=16, out_channel=32)  # [B x 32 x 4096][B x 32 x 2048]
        self.resblock2 = ResBlock1D(in_channel=32, out_channel=64)  # [B x 64 x 1024][B x 64 x 512]
        self.resblock3 = ResBlock1D(in_channel=64, out_channel=128)  # [B x 128 x 256][B x 128 x 128]
        self.resblock4 = ResBlock1D(in_channel=128, out_channel=256)  # [B x 256 x 64][B x 256 x 32]
        self.resblock5 = ResBlock1D(in_channel=256, out_channel=512)  # [B x 512 x 16][B x 512 x 8]

        # decoder

        self.up_without_z = nn.ConvTranspose1d(in_channels=512,
                                               out_channels=512,
                                               kernel_size=32,
                                               stride=2,
                                               padding=15)
        self.bn_without_z = nn.BatchNorm1d(512)
        self.prelu_without_z = nn.PReLU()

        self.up2 = ResUpSample1D(in_channel=1024, out_channel=256)

        self.up3 = ResUpSample1D(in_channel=512, out_channel=256)
        self.up4 = ResUpSample1D(in_channel=512, out_channel=128)

        self.up5 = ResUpSample1D(in_channel=256, out_channel=128)
        self.up6 = ResUpSample1D(in_channel=256, out_channel=64)

        self.up7 = ResUpSample1D(in_channel=128, out_channel=64)
        self.up8 = ResUpSample1D(in_channel=128, out_channel=32)

        self.up9 = ResUpSample1D(in_channel=64, out_channel=32)
        self.up10 = ResUpSample1D(in_channel=64, out_channel=16)

        self.final_transconv = nn.ConvTranspose1d(in_channels=16,
                                                  out_channels=1,
                                                  kernel_size=32,
                                                  stride=2,
                                                  padding=15)
        self.final_activation = nn.Tanh()

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
        """
        :param x: noisy signal
        :param z: random Gaussian noise with size 1024 x 8
        :return:
        """
        x = self.in_conv(x)
        x = self.in_bn(x)
        e0 = self.in_prelu(x)  # e0 16 x 8192

        e1, e2 = self.resblock1(e0)  # e1 32 x 4096 e2 32 x 2048
        e3, e4 = self.resblock2(e2)  # e3 64 x 1024 e4 64 x 512
        e5, e6 = self.resblock3(e4)  # e5 128 x 256 e6 128 x 128
        e7, e8 = self.resblock4(e6)  # e7 256 x 64 e8 256 x 32
        e9, e10 = self.resblock5(e8)  # e9 512 x 16 e10 512 x 8

        d1 = self.up_without_z(e10)
        d1 = self.bn_without_z(d1)
        d1 = self.prelu_without_z(d1)  # in:512 x 8 -> d1 512 x 16

        d2 = self.up2(d1, e9)  # in:1024 x 16 -> d2 256 x 32 -> cat e8
        d3 = self.up3(d2, e8)  # in:512 x 32 -> d3 256 x 64 -> cat e7
        d4 = self.up4(d3, e7)  # in:512 x 64 -> d4 128 x 128 -> cat e6
        d5 = self.up5(d4, e6)  # in:256 x 128 -> d5 128 x 256 -> cat e5
        d6 = self.up6(d5, e5)  # in:256 x 256 -> d6 64 x 512 -> cat e4
        d7 = self.up7(d6, e4)  # in:128 x 512 -> d7 64 x 1024 ->cat e3
        d8 = self.up8(d7, e3)   # in:128 x 1024 -> d8 32 x 2048 -> cat e2
        d9 = self.up9(d8, e2)  # in:64 x 2048 -> d9 32 x 4096 -> cat e1
        d10 = self.up10(d9, e1)  # in:64 x 4096 -> d10 16 x 8192 cat e0
        d11 = self.final_transconv(d10)  # in:32 x 8192 -> d11 1 x 16384

        out = self.final_activation(d11)  # Tanh()
        return out


class DiscriBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DiscriBlock, self).__init__()
        self.downsample = nn.Conv1d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=4)

        self.conv1 = nn.Conv1d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=31,
                               stride=2,
                               padding=15)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.lkrelu1 = nn.LeakyReLU(0.03)

        self.conv2 = nn.Conv1d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=31,
                               stride=2,
                               padding=15)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.lkrelu2 = nn.LeakyReLU(0.03)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lkrelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.lkrelu2(out)
        return out


class ResDiscriminator(nn.Module):
    def __init__(self):
        super(ResDiscriminator, self).__init__()
        self.in_conv = nn.Conv1d(in_channels=2,
                                 out_channels=16,
                                 kernel_size=31,
                                 stride=2,
                                 padding=15)
        self.in_bn = nn.BatchNorm1d(16)
        self.in_lkrelu = nn.LeakyReLU(0.03)  # 8192

        self.layer1 = DiscriBlock(16, 32)  # 8192->2048
        self.layer2 = DiscriBlock(32, 64)  # 2048->512
        self.layer3 = DiscriBlock(64, 128)  # 512->128
        self.layer4 = DiscriBlock(128, 256)  # 128->32
        self.layer5 = DiscriBlock(256, 512)  # 32->8

        self.final_conv1 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=8)  # B x 1 x 1

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_bn(x)
        x = self.in_lkrelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.final_conv1(x)
        x = torch.flatten(x, 1)
        return x


class ResBlock2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock2D, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                                  out_channels=out_channel,
                                                  kernel_size=1,
                                                  stride=4),
                                        nn.BatchNorm2d(num_features=out_channel))

        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=5,
                               stride=2,
                               padding=2)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=5,
                               stride=2,
                               padding=2)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        out1 = self.relu1(x)
        x = self.conv2(out1)
        x = self.bn2(x)
        x += identity
        out2 = self.relu2(x)
        return out1, out2


class ResUpSample2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResUpSample2D, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels=in_channel,
                                            out_channels=out_channel,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=1)
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LatentConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LatentConvLayer, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=in_channel,
                                 out_channels=out_channel,
                                 kernel_size=5,
                                 stride=2,
                                 padding=2)
        self.in_bn = nn.BatchNorm2d(num_features=out_channel)
        self.in_relu = nn.ReLU()

    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_bn(x)
        x = self.in_relu(x)
        return x


class LatentTransConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LatentTransConvLayer, self).__init__()
        self.in_transconv = nn.ConvTranspose2d(in_channels=in_channel,
                                               out_channels=out_channel,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1)
        self.in_bn = nn.BatchNorm2d(num_features=out_channel)
        self.in_relu = nn.ReLU()

    def forward(self, x):
        x = self.in_transconv(x)
        x = self.in_bn(x)
        x = self.in_relu(x)
        return x


# A GAN using spectrogram as input features but not signal wave form
class ResGenerator2D(nn.Module):
    def __init__(self):
        super(ResGenerator2D, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 2), stride=1)
        self.in_bn = nn.BatchNorm2d(num_features=16)
        self.in_relu = nn.ReLU()

        self.resblock1 = ResBlock2D(in_channel=16, out_channel=32)
        self.resblock2 = ResBlock2D(in_channel=32, out_channel=64)
        self.resblock3 = ResBlock2D(in_channel=64, out_channel=128)

        self.latent_conv = LatentConvLayer(in_channel=128, out_channel=256)
        self.latent_transconv = LatentTransConvLayer(in_channel=256, out_channel=128)

        self.up1 = ResUpSample2D(in_channel=256, out_channel=128)
        self.up2 = ResUpSample2D(in_channel=256, out_channel=64)

        self.up3 = ResUpSample2D(in_channel=128, out_channel=64)
        self.up4 = ResUpSample2D(in_channel=128, out_channel=32)

        self.up5 = ResUpSample2D(in_channel=64, out_channel=32)
        self.up6 = ResUpSample2D(in_channel=64, out_channel=16)

        self.final_transconv = nn.ConvTranspose2d(in_channels=32,
                                                  out_channels=32,
                                                  kernel_size=(2, 2),
                                                  stride=1)
        self.final_bn = nn.BatchNorm2d(num_features=32)
        self.final_relu = nn.ReLU()

        self.final_downsample = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.final_activation = nn.LogSigmoid()

        self.init_weights()

    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_bn(x)
        e0 = self.in_relu(x)  # e0 16 x 256 x 1024

        e1, e2 = self.resblock1(e0)  # e1 32 x 128 x 512; e2 32 x 64 x 256
        e3, e4 = self.resblock2(e2)  # e3 64 x 32 x 128; e4 64 x 16 x 64
        e5, e6 = self.resblock3(e4)  # e5 128 x 8 x 32; e6 128 x 4 x 16
        e7 = self.latent_conv(e6)  # e7 256 x 2 x 8;

        d1 = self.latent_transconv(e7)  # d1 128 x 4 x 16
        d2 = self.up1(d1, e6)  # d2 128 x 8 x 32
        d3 = self.up2(d2, e5)  # d3 64 x 16 x 64
        d4 = self.up3(d3, e4)  # d4 64 x 32 x 128
        d5 = self.up4(d4, e3)  # d5 32 x 64 x 256
        d6 = self.up5(d5, e2)  # d6 32 x 128 x 512
        d7 = self.up6(d6, e1)  # d7 16 x 256 x 1024

        out = self.final_transconv(torch.cat((d7, e0), dim=1))  # out 16 x 257 x 1025
        out = self.final_bn(out)
        out = self.final_relu(out)
        out = self.final_downsample(out)
        out = self.final_activation(out)
        return out

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResDiscriminator2D(nn.Module):
    def __init__(self):
        super(ResDiscriminator2D, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(2, 2), stride=1)
        self.in_bn = nn.BatchNorm2d(num_features=32)
        self.in_relu = nn.ReLU()

        self.resblock1 = ResBlock2D(in_channel=32, out_channel=64)
        self.resblock2 = ResBlock2D(in_channel=64, out_channel=128)
        self.resblock3 = ResBlock2D(in_channel=128, out_channel=256)
        self.latent_conv = LatentConvLayer(in_channel=256, out_channel=256)

        self.final_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(2, 8))
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_bn(x)
        x = self.in_relu(x)  # 32 x 256 x 1024

        _, x = self.resblock1(x)  # 64 x 64 x 256
        _, x = self.resblock2(x)  # 128 x 16 x 64
        _, x = self.resblock3(x)  # 64 x 4 x 16

        x = self.latent_conv(x)  # 32 x 2 x 8
        x = self.final_conv(x)
        x = torch.flatten(x, start_dim=1)
        return x


if __name__ == "__main__":
    m = ResDiscriminator2D()
    t = torch.randn(2, 2, 257, 1025)
    out = m(t)
    print(out.size())
    
