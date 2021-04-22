from collections import deque
from random import sample

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)

    def add(self, episode):
        self.buffer.append(episode)

    def replay(self, batch_size=10):
        batch = sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, max_size=100):
        super().__init__(max_size=max_size)
        self.priority = np.zeros(max_size, dtype=np.float32)
        self.priority[0] = 1.0

    def add(self, episode):
        if len(self.buffer) < self.buffer.maxlen:
            self.priority[len(self.buffer)] = self.priority.max()
            self.buffer.append(episode)
            return

        self.buffer.append(episode)

        self.priority[0:-1] = self.priority[1:]
        self.priority[-1] = self.priority.max()

    def replay(self, batch_size, alpha=0.6, beta=0.4):
        priorities = self.priority[:len(self.buffer)]
        priorities = priorities ** alpha
        prob = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), size=batch_size, p=prob)

        weights = (self.buffer.maxlen * prob[indices]) ** (-beta)
        weights = weights / weights.max()

        return [self.buffer[i] for i in indices], torch.from_numpy(indices), torch.from_numpy(weights)

    def update_priority(self, indices, priority):
        self.priority[indices] = priority + 1e-4

