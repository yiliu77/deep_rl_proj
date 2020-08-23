import gym
import numpy as np


class BrokenJointEnv(gym.Wrapper):
    def __init__(self, env, broken_joints):
        super(BrokenJointEnv, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            dtype=np.float32,
        )
        self._max_episode_steps = env._max_episode_steps
        if broken_joints is not None:
            for broken_joint in broken_joints:
                assert 0 <= broken_joint <= len(self.action_space.low) - 1
        self.broken_joints = broken_joints

    def step(self, action: np.ndarray):
        action = np.array(action)
        if self.broken_joints is not None:
            for broken_joint in self.broken_joints:
                action[broken_joint] = 0
        return super(BrokenJointEnv, self).step(action)
