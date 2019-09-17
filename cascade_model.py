import torch
import torch.nn as nn
import types


# TODO complete cascade 
class CascadeDownBlock(nn.Module):
    def __init__(self, in_plane, out_plane, use_residual=False):
        super(CascadeDownBlock, self).__init__()
        self.use_residual = use_residual
        self.downsample = nn.Conv1d(in_channels=in_plane,
                                    out_channels=out_plane,
                                    kernel_size=1,
                                    stride=2)
        self.conv = nn.Conv1d(in_channels=in_plane,
                              out_channels=out_plane,
                              kernel_size=32,
                              stride=2,
                              padding=15)
        self.bn = nn.BatchNorm1d(out_plane)
        self.prelu = nn.PReLU()

    def forward(self, x):
        if self.use_residual:
            identity = self.downsample(x)
            x = self.conv(x)
            x = self.bn(x)
            x += identity
            x = self.prelu(x)
            return x
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.prelu(x)
            return x


class CascadeUpBlock(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(CascadeUpBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose1d(in_channels=in_plane,
                                             out_channels=out_plane,
                                             kernel_size=32,
                                             stride=2,
                                             padding=15)
        self.bn = nn.BatchNorm1d(out_plane)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.trans_conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


# implement cascade learning on the denosing auto encoder
class CascadeAutoEncoder(nn.Module):
    def __init__(self, device):
        super(CascadeAutoEncoder, self).__init__()
        # when a new layer is added we need to move this layer to the device
        self.device = device
        # a container that contain nn.Module or None
        self.layer_container = [None for _ in range(20)]

        self.in_conv = nn.Conv1d(in_channels=1,
                                 out_channels=16,
                                 kernel_size=32,
                                 stride=2,
                                 padding=15)
        self.in_bn = nn.BatchNorm1d(16)
        self.in_prelu = nn.PReLU()

        self.layers = [l for l in self.layer_container if l is not None]
        self.hidden_pipeline = nn.Sequential(*self.layers)

        self.out_trans_conv = nn.ConvTranspose1d(in_channels=32,
                                                 out_channels=1,
                                                 kernel_size=32,
                                                 stride=2,
                                                 padding=15)
        self.out_tanh = nn.Tanh()

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

    @staticmethod
    def init_layer(layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return layer

    @staticmethod
    def customized_forward(self, x, z):
        """

        This method override the forward() method of self.hidden_pipeline(an nn.Sequential object)

        """
        # outputs of encode layer
        enc_out_list = []
        # a flag used for append noise into enc_out_list
        flag = True
        for module in self._modules.values():
            if isinstance(module, CascadeDownBlock):
                x = module(x)
                enc_out_list.append(x)

            elif isinstance(module, CascadeUpBlock) and flag:
                x = module(torch.cat((x, z), dim=1))
                enc_out_list.pop(-1)
                flag = False

            elif isinstance(module, CascadeUpBlock):
                x = module(torch.cat((x, enc_out_list[-1]), dim=1))
                enc_out_list.pop(-1)
        return x

    def add_layer(self, enc_in, enc_out, dec_in, dec_out, layer_index):
        """
        Add layer to the model during training
        :param enc_in: in_channel of encoder layer
        :param enc_out: out_channel of encoder layer
        :param dec_in: in_channel of decoder layer
        :param dec_out: out_channel of decoder layer
        :param layer_index: index of layer waited for inserting to model
        :return:
        """
        # if None, do not add any layer
        if enc_in is None:
            print('No layer is added!')
            return
        else:
            # TODO freeze grads of converged layers
            # freeze the gradient of trained layers first
            # maximum index of self.layer_container
            max_position = 19
            current_position = layer_index - 1
            # add encode layer
            enc_layer = CascadeDownBlock(enc_in, enc_out)
            enc_layer = self.init_layer(enc_layer)
            self.layer_container[current_position] = enc_layer
            # add decode layer
            dec_layer = CascadeUpBlock(dec_in, dec_out)
            dec_layer = self.init_layer(dec_layer)
            # the encode & decode layer is symmetry
            self.layer_container[max_position - current_position] = dec_layer
            # update self.hidden_pipeline
            self.layers = [l for l in self.layer_container if l is not None]
            # override the hidden_pipeline and move it to the device!
            self.hidden_pipeline = nn.Sequential(*self.layers).to(self.device)
            # override this nn.Sequential object's forward method
            self.hidden_pipeline.forward = types.MethodType(self.customized_forward, self.hidden_pipeline)
            return

    def forward(self, x, z):
        """
        Forward pass of generator.

        Args:
            x: input batch (signal)
            z: latent vector a Gaussian noise
        """
        # encoding step
        x = self.in_conv(x)
        x = self.in_bn(x)
        x = self.in_prelu(x)  # 16 x 8192
        begin_identity = x
        # at start there is no hidden layers
        # only train the in & out convolution layers
        if not self.layers:
            x = torch.cat((x, z), dim=1)
            x = self.out_trans_conv(x)  # 1 x 16384
            x = self.out_tanh(x)
            return x
        # if there exist hidden convolution layers
        else:
            x = self.hidden_pipeline(x, z)
            x = torch.cat((x, begin_identity), dim=1)
            x = self.out_trans_conv(x)  # 1 x 16384
            x = self.out_tanh(x)
            return x

