from buffer import ReplayBuffer


class RDPG:
    def __init__(self, buffer_size=100):
        self.buffer = ReplayBuffer(buffer_size)

    def store_episode(self, episode):
        self.buffer.add(episode)

