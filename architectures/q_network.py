import torch
from torch import nn
from architectures.utils import MLP, CNN


class ContQNet(nn.Module):
    def __init__(self, n_states, n_actions, architecture):
        super().__init__()
        self.cnn_first = "cnn_channels" in architecture
        if self.cnn_first:
            self.cnn = CNN(architecture["cnn_channels"], architecture["cnn_kernels"], architecture["cnn_strides"],
                           architecture["cnn_activation"])
            mlp_input_dim = architecture["cnn_output_dim"]
        else:
            mlp_input_dim = n_states

        layers = architecture["mlp_layers"]
        activation = architecture["mlp_activation"]
        self.model = MLP([mlp_input_dim + n_actions] + layers + [1], activation)

    def forward(self, states, actions):
        if self.cnn_first:
            states = torch.flatten(self.cnn(states), start_dim=1)
        return self.model(torch.cat([states, actions], 1))


class DiscreteQNet(nn.Module):
    def __init__(self, n_states, n_actions, architecture):
        super().__init__()
        self.cnn_first = "cnn_channels" in architecture
        if self.cnn_first:
            self.cnn = CNN(architecture["cnn_channels"], architecture["cnn_kernels"], architecture["cnn_strides"],
                           architecture["cnn_activation"])
            mlp_input_dim = architecture["cnn_output_dim"]
        else:
            mlp_input_dim = n_states

        layers = architecture["mlp_layers"]
        activation = architecture["mlp_activation"]
        self.model = MLP([mlp_input_dim] + layers + [n_actions], activation)

    def forward(self, states):
        if self.cnn_first:
            states = torch.flatten(self.cnn(states), start_dim=1)
        return self.model(states)


class TwinQNet(nn.Module):
    def __init__(self, n_states, n_actions, architecture):
        super().__init__()
        self.cnn_first = "cnn_channels" in architecture
        if self.cnn_first:
            self.cnn = CNN(architecture["cnn_channels"], architecture["cnn_kernels"], architecture["cnn_strides"],
                           architecture["cnn_activation"])
            mlp_input_dim = architecture["cnn_output_dim"]
        else:
            mlp_input_dim = n_states

        q_architecture = {
            "mlp_layers": architecture["mlp_layers"],
            "mlp_activation": architecture["mlp_activation"]
        }
        self.q_net1 = ContQNet(mlp_input_dim, n_actions, q_architecture)
        self.q_net2 = ContQNet(mlp_input_dim, n_actions, q_architecture)

    def forward(self, states, actions):
        if self.cnn_first:
            states = torch.flatten(self.cnn(states), start_dim=1)
        return self.q_net1(states, actions), self.q_net2(states, actions)
