from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)

    def add(self, episode):
        self.buffer.append(episode)

