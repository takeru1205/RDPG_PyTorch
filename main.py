import gym


env = gym.make('Pendulum-v0')

buffer = ReplayBuffer()

for e in range(5):
    cumulative_reward = 0
    obs = env.reset()
    obs_seq = [obs]
    action_seq = []
    reward_seq = []
    info_seq = []
    for t in range(env.spec.max_episode_steps):
        action = env.action_space.sample()

        new_obs, reward, info, _ = env.step(action)
        obs_seq.append(new_obs)
        action_seq.append(action)
        reward_seq.append(reward)
        info_seq.append(info)

        cumulative_reward += reward

    print(cumulative_reward)
    buffer.add([obs_seq, action_seq, reward_seq, info_seq])



