import os

import numpy as np
import torch
from torch.nn import functional
from torch.optim import Adam

from architectures.gaussian_policy import GaussianPolicy
from architectures.q_network import TwinQNet
from architectures.utils import polyak_update


class SAC:
    def __init__(self, state_dim, action_dim, action_range, ent_adj, alpha, gamma, lr, tau, n_until_target_update,
                 device, architecture):
        self.device = device

        self.policy = GaussianPolicy(state_dim, action_dim, action_range, architecture).to(self.device)
        self.policy_opt = Adam(self.policy.parameters(), lr=lr)

        self.twin_q = TwinQNet(state_dim, action_dim, architecture).to(self.device)
        self.twin_q_opt = Adam(self.twin_q.parameters(), lr=lr)
        self.target_twin_q = TwinQNet(state_dim, action_dim, architecture).to(self.device)
        polyak_update(self.twin_q, self.target_twin_q, 1)

        self.tau = tau
        self.gamma = gamma
        self.n_until_target_update = n_until_target_update

        self.alpha = alpha
        self.ent_adj = ent_adj
        if ent_adj:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = Adam([self.log_alpha], lr=lr)

        self.steps = 0

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.as_tensor(state[np.newaxis, :].copy(), dtype=torch.float32).to(self.device)
            if deterministic:
                _, _, action = self.policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]

    def update(self, states, actions, rewards, next_states, ends):
        states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.as_tensor(rewards[:, np.newaxis], dtype=torch.float32).to(self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
        ends = torch.as_tensor(ends[:, np.newaxis], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_s_action, next_s_log_prob, _ = self.policy.sample(next_states)
            next_s_q1, next_s_q2 = self.target_twin_q(next_states, next_s_action)
            next_s_q = torch.min(next_s_q1, next_s_q2)

            v = next_s_q - self.alpha * next_s_log_prob
            expected_q = rewards + (torch.ones_like(ends, requires_grad=False) - ends) * self.gamma * v

        # Q backprop
        pred_q1, pred_q2 = self.twin_q(states, actions)
        q_loss = functional.mse_loss(pred_q1, expected_q) + functional.mse_loss(pred_q2, expected_q)

        self.twin_q_opt.zero_grad()
        q_loss.backward()
        self.twin_q_opt.step()

        # Policy backprop
        s_action, s_log_prob, _ = self.policy.sample(states)
        policy_pred_q1, policy_pred_q2 = self.twin_q(states, s_action)
        policy_loss = self.alpha * s_log_prob - torch.min(policy_pred_q1, policy_pred_q2)
        policy_loss = policy_loss.mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        if self.ent_adj:
            alpha_loss = -(self.log_alpha * (s_log_prob + self.target_entropy).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.alpha = self.log_alpha.exp()

        self.steps += 1
        if self.steps % self.n_until_target_update == 0:
            polyak_update(self.twin_q, self.target_twin_q, self.tau)
            self.steps = 0

        q_val = torch.min(pred_q1 + pred_q2).mean()
        q_next = next_s_q.mean()
        if self.ent_adj:
            return q_val, q_next, self.alpha.item(), q_loss, policy_loss
        else:
            return q_val, q_next, self.alpha, q_loss, policy_loss

    def save_model(self, folder_name):
        path = 'saved_weights/' + folder_name
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), path + '/policy')
        torch.save(self.twin_q.state_dict(), path + '/twin_q_net')

    # Load model parameters
    def load_model(self, folder_name, device):
        path = 'saved_weights/' + folder_name
        self.policy.load_state_dict(torch.load(path + '/policy', map_location=torch.device(device)))
        self.twin_q.load_state_dict(torch.load(path + '/twin_q_net', map_location=torch.device(device)))

        polyak_update(self.twin_q, self.target_twin_q, 1)
        polyak_update(self.twin_q, self.target_twin_q, 1)
