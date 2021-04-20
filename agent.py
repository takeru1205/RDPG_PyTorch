import torch

from buffer import ReplayBuffer
from model import Actor, Crtic


class RDPG:
    def __init__(self, obs_dim, action_dim, buffer_size=100):
        self.buffer = ReplayBuffer(buffer_size)
        self.actor = Actor(obs_dim, action_dim)
        self.target_actor = Actor(obs_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = Crtic(obs_dim, action_dim)
        self.target_critic = Crtic(obs_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def store_episode(self, episode):
        self.buffer.add(episode)

    def get_action(self, obs, hidden_in):
        action, hidden_out = self.actor(torch.tensor(obs).to(torch.float).reshape(1, 1, 3), hidden_in)
        return action[0, 0].detach().numpy(), hidden_out

    def update(self, batch_size=10):
        if len(self.buffer) < batch_size:
            return 
        batch = self.buffer.replay(batch_size=batch_size)

        obs_batch, action_batch, reward_batch, done_batch = [], [], [], []
        for episode in batch:
            obs_batch.append(episode[0])
            action_batch.append(episode[1])
            reward_batch.append(episode[2])
            done_batch.append(episode[3])

        obs_tensor = torch.cat(obs_batch).reshape(batch_size, *obs_batch[0].shape[1:])  # Shape(batch_size, episode_length+1, 3)
        next_obs_tensor = obs_tensor[:, 1: :]  # Shape(batch_size, episode_length, 3)
        obs_tensor = obs_tensor[:, :-1, :]  # Shape(batch_size, episode_length, 3)
        action_tensor = torch.FloatTensor(action_batch)  # Shape(batch_size, episode_length, 1)
        reward_tensor = torch.FloatTensor(reward_batch)  # Shape(batch_size, episode_length, 1)
        done_tensor = torch.FloatTensor(done_batch)  # Shape(batch_size, episode_length, 1)


        hidden = (torch.randn(1, batch_size, 64),
                  torch.randn(1, batch_size, 64))  # Shape(1, batch_size, hidden_size)

        with torch.no_grad():
            target_q, _ = self.target_critic(torch.cat([obs_tensor, action_tensor], dim=2), hidden)  # Shape(batch_size, episode_length, 1)

