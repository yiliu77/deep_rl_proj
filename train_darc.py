import gym

from models.darc import DARC
from environments.broken_joint import BrokenJointEnv

source_env = gym.make('Ant-v2')
target_env = BrokenJointEnv(gym.make('Ant-v2'), [0, 1, 2, 3, 4, 5])
# env._max_episode_steps = 3000
state_dim = source_env.observation_space.shape[0]
action_dim = source_env.action_space.shape[0]

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
sa_config = {
    "input_dim": [state_dim + action_dim],
    "architecture": [{"name": "linear1", "size": 64},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": "relu",
    "output_activation": "none"
}
sas_config = {
    "input_dim": [state_dim * 2 + action_dim],
    "architecture": [{"name": "linear1", "size": 64},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": "relu",
    "output_activation": "none"
}
model = DARC(policy_config, value_config, sa_config, sas_config, source_env, target_env, "cuda", ent_adj=True,
             n_updates_per_train=2)
# model.train(500)
# model.load_model("Ant-v2-DARC2-400", "cuda")
model.train(400, deterministic=False)
# model.save_model("Ant-v2-DARC2-400")
model.eval(100)
