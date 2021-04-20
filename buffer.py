from collections import deque
from random import sample

class ReplayBuffer:
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)

    def add(self, episode):
        self.buffer.append(episode)

    def replay(self, batch_size=10):
        if len(self.buffer) < batch_size:
            return
        batch = sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

