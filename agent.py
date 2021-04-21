import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from buffer import ReplayBuffer
from model import Actor, Crtic


class RDPG:
    def __init__(self, env, initial_act=30, gamma=0.98, tau=0.001, actor_lr=1e-4, critic_lr=1e-3, buffer_size=100):

        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.initial_act = initial_act

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.buffer = ReplayBuffer(buffer_size)
        self.actor = Actor(self.obs_dim, self.action_dim)
        self.target_actor = Actor(self.obs_dim, self.action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = Crtic(self.obs_dim, self.action_dim)
        self.target_critic = Crtic(self.obs_dim, self.action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.criterion = nn.MSELoss()

    def store_episode(self, episode):
        self.buffer.add(episode)

    def get_action(self, obs, action, hidden_in, epoch=None):
        if epoch is not None and epoch < self.initial_act:
            return self.env.action_space.sample() / self.env.action_space.high[0], None
        history = torch.cat([torch.FloatTensor(obs), torch.FloatTensor(action)]).to(torch.float).reshape(1, 1, self.obs_dim+self.action_dim)
        action, hidden_out = self.actor(history, hidden_in)
        if epoch is None:
            return action[0, 0].detach().numpy(), hidden_out
        action = action[0, 0].detach().numpy() + np.random.normal(0, 0.3)
        return np.clip(action, -1, 1), hidden_out

    def soft_update(self, target_net, net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

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
        next_action_tensor = action_tensor[:, 1:, :]
        action_tensor = action_tensor[:, :-1, :]
        reward_tensor = torch.FloatTensor(reward_batch).unsqueeze(dim=-1)  # Shape(batch_size, episode_length, 1)
        done_tensor = torch.FloatTensor(done_batch).unsqueeze(dim=-1)  # Shape(batch_size, episode_length, 1)

        hidden = (torch.randn(1, batch_size, 64),
                  torch.randn(1, batch_size, 64))  # Shape(1, batch_size, hidden_size)

        with torch.no_grad():
            target_action, _ = self.target_actor(torch.cat([next_obs_tensor, next_action_tensor], dim=2), hidden)  # Shape(batch_size, episode_length, 1)
            target_q, _ = self.target_critic(torch.cat([next_obs_tensor, target_action], dim=2), hidden)  # Shape(batch_size, episode_length, 1)
            y = reward_tensor + done_tensor * self.gamma * target_q  # Shape(batch_size, episode_length, 1)

        q_values, _ = self.critic(torch.cat([obs_tensor, action_tensor], dim=2), hidden)
        critic_loss = self.criterion(q_values, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action, _ = self.actor(torch.cat([obs_tensor, action_tensor], dim=2), hidden)
        actor_loss = self.critic(torch.cat([obs_tensor, action], dim=2), hidden)[0].mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.target_critic, self.critic)
        self.soft_update(self.target_actor, self.actor)




