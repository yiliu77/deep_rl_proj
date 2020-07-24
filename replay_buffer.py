import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, k=self.batch_size)
        return list(map(np.array, zip(*batch)))

    def __len__(self):
        return len(self.buffer)
