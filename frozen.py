"""
FrozenLake environment in gym openai

Actions
    - 0: Left
    - 1: Down 
    - 2: Right
    - 3: Up

Rewards:
    - 1: reach goal
    - 0: otherwise
"""

import gym
import time
import numpy as np
import random 

ENVIRONMENT = 'FrozenLake8x8-v0'
ALPHA = 0.4
DISCOUNT_FACTOR = 0.9
EPISODES = 5000
STEPS = 100
EPSILON = 0.05
ALL_ACTIONS = [0,1,2,3]
QTABLE = True

# Initialise 
env = gym.make(ENVIRONMENT)
Q = np.zeros([env.observation_space.n, env.action_space.n]) # Q table

# Q-learning
for eps in range(EPISODES):
    state = env.reset() # Initialise state
    #time.sleep(0.1)
    for step in range(STEPS):
        # Select the next action from Q(s,a): epsilon greedy with tie breaking
        prob = np.random.random()
        if prob <= EPSILON:
            action = random.choice(ALL_ACTIONS)
        else:
            maxQ = np.max(Q[state, :])
            actions = np.argwhere(Q[state, :] == maxQ).flatten().tolist()
            action = random.choice(actions)

        # Execute action and observe new state
        nextState, reward, done, info = env.step(action)
        
        # Update Q
        Q[state, action] += ALPHA * (reward + DISCOUNT_FACTOR * np.max(Q[nextState, :]) - Q[state, action])
        
        state = nextState

        # Check if it's time to change to next episode
        if done:
            #print("Episode {} finished after {} timesteps".format(eps+1, step+1))
            break
    #env.render()

if QTABLE:
    print (Q)