import gym
from random import random, randrange
import matplotlib.pyplot as plt
import numpy as np
env = gym.make('FrozenLake-v0')

actions = {0:0, 1:0, 2:0, 3:0, "random":0, "max":0}
max_Q = {0: [0.5283121039950627, 0.43000303101637927, 0.42869826215197193, 0.4204035112988539], 1: [0.2353997883350748, 0.0036942107923493674, 0.0007564469302984774, 0.0026718883056462035], 2: [0.014523076770726754, 0.18516836176331508, 0.020083479296916937, 0.012993953210488941], 3: [0.0730225347339686, 0.0, 0.0, 0.0], 4: [0.5466052694576714, 0.4316397372108541, 0.3009936164999034, 0.348464419356063], 5: [0.0, 0.0, 0.0, 0.0], 6: [0.31998984258987356, 0.054346360878382946, 0.03870689527467612, 0.0004941201485692323], 7: [0.0, 0.0, 0.0, 0.0], 8: [0.3357522916787141, 0.42997067176929066, 0.32223833652478184, 0.5742118627111734], 9: [0.3469184689440146, 0.6266539051631604, 0.3798081501602387, 0.31640914523459784], 10: [0.6551843527751113, 0.1743934973885426, 0.22553531496449422, 0.12159602953138665], 11: [0.0, 0.0, 0.0, 0.0], 12: [0.0, 0.0, 0.0, 0.0], 13: [0.3756902059705939, 0.3831976220059579, 0.8170857513426155, 0.4058029653800437], 14: [0.4664104042414649, 0.8979175898423615, 0.5955763439809926, 0.5384154331862244], 15: [0.0, 0.0, 0.0, 0.0]}

def select_action_epsilon(state, Q, epsilon, actions):
    if random() < 1-epsilon:
        if len(set(Q[state])) < 2:
            actions["random"] = actions["random"] + 1
            return randrange(4)
        actions["max"] = actions["max"] + 1
        a = Q[state].index(max(Q[state]))
        return a
    else:
        actions["random"] = actions["random"] + 1
        return randrange(4)

lr = 0.1
y = 0.99
episodes = 10000
Q = max_Q
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