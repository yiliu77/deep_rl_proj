import numpy as np
import gym
import cv2


class StackedEnv(gym.Wrapper):
    def __init__(self, env, width, height, n_img_stack, n_action_repeats):
        super(StackedEnv, self).__init__(env)
        self.width = width
        self.height = height
        self.n_img_stack = n_img_stack
        self.n_action_repeats = n_action_repeats
        self.stack = []

    def reset(self):
        img_rgb = super(StackedEnv, self).reset()
        img_gray = self.preprocess(img_rgb)
        self.stack = [img_gray] * self.n_img_stack
        return np.rollaxis(np.stack(self.stack, axis=2), 2, 0)

    def step(self, action):
        total_reward = 0
        done = False
        img_rgb = None
        info = None
        for i in range(self.n_action_repeats):
            img_rgb, reward, done, info = super(StackedEnv, self).step(action)
            total_reward += reward
            if done:
                break
        img_gray = self.preprocess(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.n_img_stack
        return np.rollaxis(np.stack(self.stack, axis=2), 2, 0), total_reward, done, info

    def preprocess(self, rgb_img):
        gray = np.dot(rgb_img[..., :], [0.299, 0.587, 0.114])
        gray = gray / 128. - 1.
        res = cv2.resize(gray, dsize=(self.height, self.width), interpolation=cv2.INTER_CUBIC)
        return res
