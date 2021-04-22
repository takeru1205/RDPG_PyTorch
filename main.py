import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import Actor
from agent import RDPG


env = gym.make('Pendulum-v0')

writer = SummaryWriter(log_dir="./logs")
agent = RDPG(env, writer=writer)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initial Act
print('Initial Act Sequence Start')
for e in range(30):
    obs = env.reset()
    action = env.action_space.sample() / env.action_space.high[0]
    obs_seq = torch.zeros((1, env.spec.max_episode_steps+1, 3))
    obs_seq[:, 0, :] = torch.tensor(obs)
    action_seq = [action]
    reward_seq = []
    info_seq = []
    for t in range(env.spec.max_episode_steps):
        # action = env.action_space.sample() / env.action_space.high[0]
        action = np.clip(env.action_space.sample(), -1, -1)

        new_obs, reward, info, _ = env.step(action * env.action_space.high[0])
        obs_seq[:, t+1, :] = torch.tensor(new_obs)
        action_seq.append(action)
        reward_seq.append(reward)
        info_seq.append(1 - info)

        obs = new_obs

    agent.store_episode([obs_seq, action_seq, reward_seq, info_seq])
    # agent.update()


beta_begin = 0.4
beta_end = 1.0
beta_decay = 500000
beta_func = lambda step: min(beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay))
total_step = 1

# Train
print('Train Sequence Start')
for e in range(300):
    cumulative_reward = 0
    obs = env.reset()
    action = env.action_space.sample() / env.action_space.high[0]
    obs_seq = torch.zeros((1, env.spec.max_episode_steps+1, 3))
    obs_seq[:, 0, :] = torch.tensor(obs)
    action_seq = [action]
    reward_seq = []
    info_seq = []
    hidden = (torch.randn(1, 1, 64).to(device),
              torch.randn(1, 1, 64).to(device))
    for t in range(env.spec.max_episode_steps):
        action, hidden = agent.get_action(obs, action, hidden, e, train=True)

        assert -1. <= action <= 1., f'Get {action}'

        new_obs, reward, info, _ = env.step(action * env.action_space.high[0])
        # env.render()
        obs_seq[:, t+1, :] = torch.tensor(new_obs)
        action_seq.append(action)
        reward_seq.append(reward)
        info_seq.append(1 - info)

        obs = new_obs
        cumulative_reward += reward
        total_step += 1

    print(f'Episode: {e:>3}, Reward: {cumulative_reward:>8.2f}, Avearage Action{np.array(action_seq).mean():>9.5f}')
    writer.add_scalar("Train/Reward", cumulative_reward, e)
    agent.store_episode([obs_seq, action_seq, reward_seq, info_seq])

    agent.update(e, beta=beta_func(total_step))

# Test
print('Test Sequence Start')
for e in range(5):
    cumulative_reward = 0
    obs = env.reset()
    action = env.action_space.sample() / env.action_space.high[0]
    hidden = (torch.randn(1, 1, 3).to(device),
              torch.randn(1, 1, 3).to(device))
    for t in range(env.spec.max_episode_steps):
        action, hidden = agent.get_action(obs, action, e, hidden)

        new_obs, reward, info, _ = env.step(action * env.action_space.high[0])

        obs = new_obs
        cumulative_reward += reward

    print(f'Test Episode: {e:>3}, Reward: {cumulative_reward:>8.2f}')
env.close()
