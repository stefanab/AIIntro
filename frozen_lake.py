import gym
from random import random, randrange
import matplotlib.pyplot as plt
import numpy as np
env = gym.make('FrozenLake-v0')

actions = {0:0, 1:0, 2:0, 3:0, "random":0, "max":0}
zeroes = {}
for i in range(env.observation_space.n):
    zeroes[i] = [0.0,0.0,0.0,0.0]

def select_action_epsilon(state, Q, epsilon, actions):
    if random() < 1-epsilon:
        if len(set(Q[state])) < 2:
            actions["random"] = actions["random"] + 1
            return randrange(4)
        actions["max"] = actions["max"] + 1
        a = Q[state].index(max(Q[state]))
    else:
        actions["random"] = actions["random"] + 1
        return randrange(4)

        return a
lr = 0.1
y = 0.99
episodes = 10000
Q = zeroes
win_counter = 0
death_by_slide = 0
epsilon = 0.1
average_steps = 0
rewards = []
totReward = 0

for episode in range(1,episodes+1):
    epsilon = epsilon * 0.999
    state = env.reset()

    for i in range(1000):
        # env.render()
        action = select_action_epsilon(state, Q, epsilon, actions)
        actions[action] = actions[action] + 1

        new_state, reward, done, _ = env.step(action)

        Q[state][action] = Q[state][action] + lr*(reward + y*max(Q[new_state]) - Q[state][action])
        state = new_state
        totReward += reward
        if done:
            rewards.append(totReward/episode)
            # env.render()
            average_steps = average_steps + i
            if state == 15:
                win_counter += 1
                # rewards.append(1)
            # else:
                # rewards.append(0)
            # print("Episode finished after {} timesteps".format(i+1))
            break
    # rewards.append(0)

print(average_steps)
print(average_steps/episodes)
print("Num episodes:", episodes)
print("Wins: ",win_counter)
print("Win %:", win_counter/episodes)
print(actions)
print("Final Q-Table Values")
print(Q)
print(rewards[-100:])
print(len(rewards))
plt.suptitle("Q learning")
plt.plot(rewards)
plt.show()