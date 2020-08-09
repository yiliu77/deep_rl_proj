import copy
import os

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from architectures.gaussian_policy import DiscreteGaussianPolicy
from architectures.value_networks import ValueNet
from replay_buffer import ReplayBuffer


class DiscretePPO:
    def __init__(self, policy_config, value_config, env, device, log_dir="latest_runs",
                 memory_size=1e5, warmup_iters=300, batch_size=32, lr=1e-3, gamma=0.99, lam=0.9, epsilon=0.2):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.batch_size = batch_size

        path = 'runs/' + log_dir
        if not os.path.exists(path):
            os.makedirs(path)
        self.writer = SummaryWriter(path)

        self.memory_size = memory_size
        self.warmup_iters = warmup_iters
        self.memory = ReplayBuffer(self.memory_size, self.batch_size)

        self.env = env
        self.policy = DiscreteGaussianPolicy(policy_config).to(self.device)
        self.policy_opt = Adam(self.policy.parameters(), lr=lr)

        self.val_net = ValueNet(value_config).to(self.device)
        self.val_net_opt = Adam(self.val_net.parameters(), lr=lr)

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.as_tensor(state[np.newaxis, :].copy(), dtype=torch.float32).to(self.device)
            if deterministic:
                _, action_prob, action = self.policy.sample(state)
            else:
                action, action_prob, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0], action_prob.detach().cpu().numpy()[0]

    def get_value(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state[np.newaxis, :].copy(), dtype=torch.float32).to(self.device)
            value = self.val_net(state)
            return value.detach().cpu().numpy()[0][0]

    def train_step(self, states, actions, old_action_probs, returns, values, advs):
        states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions[:, np.newaxis], dtype=torch.int64).to(self.device)
        old_action_probs = torch.as_tensor(old_action_probs, dtype=torch.float32).to(self.device)
        returns = torch.as_tensor(returns[:, np.newaxis], dtype=torch.float32).to(self.device)
        values = torch.as_tensor(values[:, np.newaxis], dtype=torch.float32).to(self.device)
        advs = torch.as_tensor(advs[:, np.newaxis], dtype=torch.float32).to(self.device)

        action_probs = self.policy(states)
        ratios = torch.exp(torch.log(action_probs.gather(1, actions)) - torch.log(old_action_probs).gather(1, actions))
        clipped_ratio = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(clipped_ratio * advs, ratios * advs).mean()

        value_preds = self.val_net(states)
        clipped_values = values + torch.clamp(value_preds - values, -self.epsilon, self.epsilon)
        value_loss1 = 0.5 * torch.square(returns - value_preds)
        value_loss2 = 0.5 * torch.square(returns - clipped_values)
        value_loss = torch.max(value_loss1, value_loss2).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        self.val_net_opt.zero_grad()
        value_loss.backward()
        self.val_net_opt.step()

        return policy_loss, value_loss, value_preds.mean()

    def gae(self, rewards, values, next_values, end_masks):
        deltas = rewards + self.gamma * next_values * end_masks - values
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]
        return gaes

    def train(self, num_games, deterministic=False):
        self.policy.train()
        self.val_net.train()
        for i in range(num_games):
            states, actions, action_probs, rewards, values, dones = [], [], [], [], [], []
            state = self.env.reset()
            done = False
            n_steps = 0
            while not done:
                if len(self.memory) <= self.warmup_iters:
                    action = self.env.action_space.sample()
                    action_prob = self.env.action_space.n * [1 / self.env.action_space.n]
                else:
                    action, action_prob = self.get_action(state, deterministic)

                value = self.get_value(state)
                next_state, reward, done, _ = self.env.step(action)
                n_steps += 1

                states.append(state)
                actions.append(action)
                action_probs.append(action_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(done if n_steps != self.env._max_episode_steps else False)

                state = next_state

            states = np.array(states)
            actions = np.array(actions)
            action_probs = np.array(action_probs)
            rewards = np.array(rewards)
            values = np.array(values)
            dones = np.array(dones)

            next_values = np.append(values[1:], [self.get_value(state)])
            done_masks = 1 - np.array(dones)
            gaes = self.gae(rewards, values, next_values, done_masks)
            returns = gaes + values
            advs = (gaes - gaes.mean()) / gaes.std()

            for point in zip(states, actions, action_probs, returns, values, advs):
                self.memory.add(*point)

            # training
            if len(self.memory) > self.warmup_iters:
                policy_losses, value_losses, value_preds = [], [], []
                for _ in range(len(states)):
                    tr_states, tr_actions, tr_old_action_probs, tr_returns, tr_values, tr_advs = self.memory.sample()
                    policy_loss, value_loss, value_pred = self.train_step(tr_states, tr_actions, tr_old_action_probs,
                                                                           tr_returns, tr_values, tr_advs)
                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    value_preds.append(value_pred)
                self.writer.add_scalar('Policy Loss', sum(policy_losses) / len(policy_losses), i)
                self.writer.add_scalar('Value Loss', sum(value_losses) / len(value_losses), i)
                self.writer.add_scalar('Reward', sum(rewards), i)

    def eval(self, num_games, render=True):
        self.policy.eval()
        self.val_net.eval()
        for i in range(num_games):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                if render:
                    env.render()
                action, action_prob = self.get_action(state, deterministic=True)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            print(i, total_reward)

    def save_model(self, folder_name):
        path = 'saved_weights/' + folder_name
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), path + '/policy')
        torch.save(self.val_net.state_dict(), path + '/val_net')

    # Load model parameters
    def load_model(self, folder_name, device):
        path = 'saved_weights/' + folder_name
        self.policy.load_state_dict(torch.load(path + '/policy', map_location=torch.device(device)))
        self.val_net.load_state_dict(torch.load(path + '/val_net', map_location=torch.device(device)))


if __name__ == "__main__":
    import gym

    env = gym.make('CartPole-v1')
    env._max_episode_steps = 3000
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy_config = {
        "input_dim": [n_states],
        "architecture": [{"name": "linear1", "size": 80},
                         {"name": "linear3", "size": n_actions}],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    value_config = {
        "input_dim": [n_states],
        "architecture": [{"name": "linear1", "size": 80},
                         {"name": "linear3", "size": 1}],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    model = DiscretePPO(policy_config, value_config, env, "cuda")

    model.load_model("CartPole-v1-PPO-5200", "cuda")
    model.train(800, deterministic=True)
    model.save_model("CartPole-v1-PPO-6000")
    model.eval(100)
