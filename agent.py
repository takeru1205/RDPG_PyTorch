import torch

from buffer import ReplayBuffer
from model import Actor


class RDPG:
    def __init__(self, obs_dim, action_dim, buffer_size=100):
        self.buffer = ReplayBuffer(buffer_size)
        self.actor = Actor(obs_dim, action_dim)

    def store_episode(self, episode):
        self.buffer.add(episode)

    def get_action(self, obs, hidden_in):
        action, hidden_out = self.actor(torch.tensor(obs).to(torch.float).reshape(1, 1, 3), hidden_in)
        return action[0, 0].detach().numpy(), hidden_out

