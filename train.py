import gym
from gym import wrappers
from torch import nn
import numpy as np

from models.sac import ContSAC
# from replay_buffer import ReplayBuffer
# from env_wrapper import StackedEnv
#
# # env_name = "CarRacing-v0"
# # env = StackedEnv(env_name, 48, 48, 4, 3)
# #
# # state_dim = env.observation_space.shape[0]
# # action_dim = env.action_space.shape[0]
# # action_range = (env.action_space.low, env.action_space.high)
#
# #### DEFINE ENVIRONMENT ####
# env_name = "Pendulum-v0"
# env = gym.make(env_name)
#
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# action_range = (env.action_space.low, env.action_space.high)
#
# # n_until_target_update = 1
# # device = "cuda"
# # ent_adj = False
# # alpha = 0.2
# # gamma = 0.99
# # lr = 0.0001
# # tau = 0.003
# # architecture = {
# #     "cnn_channels": [4, 32, 64, 64],
# #     "cnn_kernels": [8, 4, 3],
# #     "cnn_strides": [4, 2, 1],
# #     "cnn_activation": nn.ReLU(),
# #     "cnn_output_dim": 256,
# #     "mlp_layers": [256, 256],
# #     "mlp_activation": nn.ReLU()
# # }
#
# #### DEFINE MODEL ####
# device = "cuda"
#
# n_until_target_update = 1
# ent_adj = False
# alpha = 0.2
# gamma = 0.99
# lr = 0.0001
# tau = 0.003
# architecture = {
#     "mlp_layers": [256, 256],
#     "mlp_activation": nn.ReLU()
# }
#
# model = SAC(state_dim, action_dim, action_range, ent_adj, alpha, gamma, lr, tau, n_until_target_update, device,
#             architecture)
#
# capacity = 100000
# batch_size = 64
# memory = ReplayBuffer(capacity, batch_size)
#
# n_sample_epochs = 10
# n_epochs_per_train = 1
# n_steps_per_train = 200
# n_evals_per_train = 2
# n_epochs_per_save = 50
# render = False
#
# #### RESTART TRAINING ####
# start_epoch = 0
# # model.load_model("CarRacing-v0-{}".format(start_epoch), device)
#
# for epoch in range(start_epoch + 1, 100000):
#     # Gather data
#     total_rewards = 0
#     n_steps = 0
#     done = False
#     state = env.reset()
#     while not done:
#         action = model.get_action(state)
#         if epoch <= n_sample_epochs:
#             next_state, reward, done, _ = env.step(env.action_space.sample())
#         else:
#             next_state, reward, done, _ = env.step(action)
#         next_state = next_state
#         n_steps += 1
#         total_rewards += reward
#
#         end = 0 if n_steps == env._max_episode_steps else float(done)
#         memory.add(state, action, reward, next_state, end)
#
#         state = next_state
#     print("index: {}, steps: {}, total_rewards: {}".format(epoch, n_steps, total_rewards))
#
#     if epoch >= n_sample_epochs + start_epoch and epoch % n_epochs_per_train == 0:
#         # Training
#         q_vals = []
#         q_nexts = []
#         q_losses = []
#         policy_losses = []
#         alphas = []
#         for _ in range(n_steps_per_train):
#             s, a, r, s_, d = memory.sample()
#             q_val, q_next, alpha, q_loss, policy_loss = model.update(s, a, r, s_, d)
#             q_vals.append(q_val)
#             q_nexts.append(q_next)
#             alphas.append(alpha)
#             q_losses.append(q_loss)
#             policy_losses.append(policy_loss)
#         print("q_val: {}, q_next: {}, alpha: {}, q_loss: {}, policy_loss: {}"
#               .format(sum(q_vals) / len(q_vals), sum(q_nexts) / len(q_nexts), sum(alphas) / len(alphas),
#                       sum(q_losses) / len(q_losses), sum(policy_losses) / len(policy_losses)))
#
#         # Evaluation
#         all_rewards = []
#         for j in range(n_evals_per_train):
#             total_rewards = 0
#             state = env.reset()
#             done = False
#             while not done:
#                 action = model.get_action(state, deterministic=True)
#                 if j == 0 and render:
#                     env.render()
#                 next_state, reward, done, _ = env.step(action)
#                 next_state = next_state
#
#                 state = next_state
#                 total_rewards += reward
#
#             all_rewards.append(total_rewards)
#         print("avg_eval_reward: {}".format(sum(all_rewards) / len(all_rewards)))
#         print("----------------------")
#
#     if epoch % n_epochs_per_save == 0:
#         model.save_model("{}-{}".format(env_name, epoch))

if __name__ == "__main__":
    import gym

    env = gym.make('Pendulum-v0')
    # env._max_episode_steps = 3000
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy_config = {
        "input_dim": [state_dim],
        "architecture": [{"name": "linear1", "size": 256},
                         {"name": "linear2", "size": 256},
                         {"name": "split1", "sizes": [action_dim, action_dim]}],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    value_config = {
        "input_dim": [state_dim + action_dim],
        "architecture": [{"name": "linear1", "size": 256},
                         {"name": "linear2", "size": 256},
                         {"name": "linear2", "size": 1}],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    model = ContSAC(policy_config, value_config, env, "cuda")
    # model.train(500)
    # model.load_model("Pendulum-v0-SAC-800", "cuda")
    model.train(800, deterministic=False)
    model.save_model("Pendulum-v0-SAC-1600")
    model.eval(100)