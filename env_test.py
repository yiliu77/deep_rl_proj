import gym
import random

env = gym.make('Ant-v2')
state = env.reset()
done = False
total_reward = 0
while not done:
    env.render()
    next_state, reward, done, _ = env.step([random.random() * 2 - 1, random.random() * 2 - 1, 0, 0, 0, 0, 0, 0])
    total_reward += reward
    state = next_state