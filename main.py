import gym


env = gym.make('Pendulum-v0')

for e in range(5):
    cumulative_reward = 0
    obs = env.reset()
    for t in range(env.spec.max_episode_steps):
        action = env.action_space.sample()

        new_obs, reward, info, _ = env.step(action)

        cumulative_reward += reward

    print(cumulative_reward)


