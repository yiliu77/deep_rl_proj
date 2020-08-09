import gym
from torch import nn

from models.sac import SAC
from env_wrapper import StackedEnv

#### DEFINE ENVIRONMENT ####
env_name = "Pendulum-v0"
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = (env.action_space.low, env.action_space.high)

#### DEFINE MODEL ####
device = "cuda"

n_until_target_update = 1
ent_adj = False
alpha = 0.2
gamma = 0.99
lr = 0.0001
tau = 0.003
architecture = {
    "mlp_layers": [256, 256],
    "mlp_activation": nn.ReLU()
}

model = SAC(state_dim, action_dim, action_range, ent_adj, alpha, gamma, lr, tau, n_until_target_update, device,
            architecture)

n_sample_epochs = 10
n_epochs_per_train = 1
n_steps_per_train = 200
n_evals_per_train = 2
n_epochs_per_save = 50
model.load_model("Pendulum-v0-{}".format(200), device)

render = True

for epoch in range(100000):
    total_rewards = 0
    state = env.reset()
    done = False
    while not done:
        action = model.get_action(state, deterministic=True)
        print(action)
        if render:
            env.render()
        next_state, reward, done, _ = env.step(action)

        state = next_state
        total_rewards += reward
    print("total_reward: {}".format(total_rewards))