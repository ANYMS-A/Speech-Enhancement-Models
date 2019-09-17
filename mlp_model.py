import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(BasicBlock, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class SEMLP(nn.Module):

    def __init__(self, in_size, out_size, hidden_size, num_layer):
        super(SEMLP, self).__init__()
        layers = []
        if num_layer == 1:
            self.pipe_line = nn.Sequential(nn.Linear(in_size, out_size))
        else:
            layers.append(BasicBlock(in_size, hidden_size))

            for i in range(num_layer - 1):
                layers.append(BasicBlock(hidden_size, hidden_size))

            layers.append(nn.Linear(hidden_size, out_size))
            layers.append(nn.LogSigmoid())

            self.pipe_line = nn.Sequential(*layers)

        self.init_weights()

    def forward(self, x):
        x = self.pipe_line(x)
        return x

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
