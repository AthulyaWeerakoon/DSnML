import gym
import numpy as np
import random

env = gym.make("CartPole-v1")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
STATE_LOWS = env.observation_space.low
STATE_HIGHS = env.observation_space.high


def sigmoid(x: int):
    return 1 / (1 + np.exp(-x))


def make_discrete(low: float, high: float, div_count: int, val: float):
    div_size = (high - low) / div_count
    return int((val - low)/div_size)


def get_discrete_from_state(_state: tuple):
    _state = [make_discrete(STATE_LOWS[0], STATE_HIGHS[0], 32, _state[0][0]),
              make_discrete(0, 1, 32, sigmoid(_state[0][1])),
              make_discrete(STATE_LOWS[2], STATE_HIGHS[2], 32, _state[0][2]),
              make_discrete(0, 1, 32, sigmoid(_state[0][3]))]
    return tuple(_state)


env.reset()

cell_shape = [32] * len(env.observation_space.high)
cell_shape.append(2)
q_table = np.zeros(shape=cell_shape)

try:
    with open('test.npy', 'rb') as f:
        q_table = np.load(f)

except:
    print("Couldn't load file")

print(q_table.dtype)
running_reward = 0
episode_count = 0
opened_env = False

while True:
    discrete_state = get_discrete_from_state(env.reset(seed=42))
    episode_reward = 0
    while True:
        action = np.argmax(q_table[discrete_state]) if random.randint(0, 99) < 80 else np.argmin(q_table[discrete_state])
        new_state, reward, done, _, _ = env.step(action)
        next_discrete_state = get_discrete_from_state((new_state, ))
        if running_reward > 450 and not opened_env:
            env.close()
            env = gym.make("CartPole-v0", render_mode="human")
            env.reset(seed=42)
            opened_env = True
            with open('test.npy', 'wb') as f:
                np.save(f, q_table)
        if not done:
            next_q = np.max(q_table[next_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * next_q)
            q_table[discrete_state + (action, )] = new_q
            episode_reward += reward
        elif episode_reward >= 474:
            q_table[discrete_state + (action, )] = 1

        discrete_state = next_discrete_state

        if done:
            break

    episode_count += 1
    running_reward = 0.05 * episode_reward + 0.95 * running_reward

    if episode_count % 1000 == 0:
        print("Running reward: {:.2f} at episode {}".format(running_reward, episode_count))

    if running_reward > 474:
        print("Solved at episode {}!".format(episode_count))
        break


env.close()
