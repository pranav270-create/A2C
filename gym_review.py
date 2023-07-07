import gym
import numpy as np
import matplotlib.pyplot as plt

def get_action(s, w):
    return 1 if np.dot(s, w) > 0 else 0

def play_one_episode(env, params):
    observation = env.reset()[0]
    done = False
    t = 0
    r = 0
    while not done and t < 10000:
        t += 1
        action = get_action(observation, params)
        ret = env.step(action)
        observation = ret[0]
        reward = ret[1]
        done = ret[2]
        info = ret[3]
        r += reward
    return r

def play_multiple_episodes(env, T, params):
    rewards = np.empty(T)
    for i in range(T):
        rewards[i] = play_one_episode(env, params)
    avg_reward = rewards.mean()
    print("avg reward:", avg_reward)
    return avg_reward

def random_search(env):
    episode_rewards = []
    best_params = None
    best_reward = 0
    for t in range(100):
        params = np.random.random(4) * 2 - 1
        avg_reward = play_multiple_episodes(env, 100, params)
        episode_rewards.append(avg_reward)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = params
    return episode_rewards, best_params

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    episode_rewards, best_params = random_search(env)
    plt.plot(episode_rewards)
    plt.show()

    # play a final set of episodes
    print("***Final run with final weights***")
    play_multiple_episodes(env, 100, best_params)