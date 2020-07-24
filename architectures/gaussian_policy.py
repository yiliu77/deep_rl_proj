import torch
from torch import nn, distributions
from architectures.utils import MLP, CNN


class GaussianPolicy(nn.Module):
    def __init__(self, n_states, n_actions, action_range, architecture):
        super(GaussianPolicy, self).__init__()
        self.cnn_first = "cnn_channels" in architecture
        if self.cnn_first:
            self.cnn = CNN(architecture["cnn_channels"], architecture["cnn_kernels"], architecture["cnn_strides"],
                           architecture["cnn_activation"])
            mlp_input_dim = architecture["cnn_output_dim"]
        else:
            mlp_input_dim = n_states

        layers = architecture["mlp_layers"]
        activation = architecture["mlp_activation"]

        self.mlp = MLP([mlp_input_dim] + layers, activation)
        self.mu = nn.Linear(layers[-1], n_actions)
        self.log_std = nn.Linear(layers[-1], n_actions)

        action_low, action_high = action_range
        self.action_scale = torch.as_tensor((action_high - action_low) / 2, dtype=torch.float32)
        self.action_bias = torch.as_tensor((action_high + action_low) / 2, dtype=torch.float32)

    def forward(self, states):
        if self.cnn_first:
            states = torch.flatten(self.cnn(states), start_dim=1)
        x = self.mlp(states)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        return mu, log_std

    def sample(self, states):
        mu, log_std = self.forward(states)
        std = torch.exp(log_std)

        normal_dist = distributions.Normal(mu, std)
        output = normal_dist.rsample()
        tanh_output = torch.tanh(output)
        action = self.action_scale * tanh_output + self.action_bias
        mean_action = self.action_scale * torch.tanh(mu) + self.action_bias

        log_prob = normal_dist.log_prob(output)
        # https://arxiv.org/pdf/1801.01290.pdf appendix C
        log_prob -= torch.log(
            self.action_scale * (torch.ones_like(tanh_output, requires_grad=False) - tanh_output.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean_action

    def to(self, *args, **kwargs):
        device = args[0]
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
