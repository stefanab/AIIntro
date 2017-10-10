import gym
import random

env = gym.make("FrozenLake-v0")
actionValue = [0.5,1,0.5,0.5]
states = []
epsilon = 0.1
for x in range(16):
    values = []
    for y in range(4):
        values.append(actionValue[y])
        
    states.append(values)
    
def qfunc(state):
    currentActionValues = states[state]
    highest = 0
    spot = 0
    for a in range (1,len(currentActionValues)):
        if currentActionValues[a] > highest:
            highest = currentActionValues[a]
            spot = a
    
    if(highest == 0):	
        return random.randint(0,3)
    return spot


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = qfunc(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    epsilon = epsilon * 0.999