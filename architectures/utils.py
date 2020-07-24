from torch import nn


def polyak_update(network, target_network, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))


class MLP(nn.Module):
    def __init__(self, layers, activation=nn.ReLU):
        super().__init__()

        weights = []
        for i in range(len(layers) - 1):
            weights.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                weights.append(activation)
        self.model = nn.Sequential(*weights)

    def forward(self, inputs):
        return self.model(inputs)


class CNN(nn.Module):
    def __init__(self, channels, kernels, strides, activation=nn.ReLU):
        super().__init__()
        assert len(channels) - 1 == len(kernels) == len(strides)

        weights = []
        for i in range(len(channels) - 1):
            weights.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i + 1],
                                     kernel_size=kernels[i], stride=strides[i]))
            weights.append(activation)
        self.model = nn.Sequential(*weights)

    def forward(self, inputs):
        return self.model(inputs)