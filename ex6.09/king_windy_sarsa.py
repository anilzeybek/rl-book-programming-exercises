from king_windy_gridworld_env import KingWindyGridWorldEnv
import numpy as np

env = KingWindyGridWorldEnv()
env.reset()

ALPHA = 0.5
GAMMA = 1
EPSILON = 0.1
STEP_SIZE = 10000

number_of_states = env.grid_height * env.grid_width
number_of_actions = len(env.actions)


def initialize_Q():
    Q = {}
    for i in range(env.grid_height):
        for j in range(env.grid_width):
            for a in range(number_of_actions):
                Q[(i, j)] = [0] * number_of_actions

    return Q


Q_table = initialize_Q()

for _ in range(STEP_SIZE):
    env.reset()
    current_state = env.observation
    episode_ended = False

    if EPSILON >= np.random.uniform():
        action = np.random.randint(number_of_actions)
    else:
        action = np.argmax(Q_table[current_state])

    while True:
        new_state, reward, episode_ended, _ = env.step(action)
        if episode_ended:
            break

        if EPSILON >= np.random.uniform():
            new_action = np.random.randint(number_of_actions)
        else:
            new_action = np.argmax(Q_table[new_state])

        Q_table[current_state][action] += ALPHA * (reward + GAMMA * Q_table[new_state][new_action] -
                                                   Q_table[current_state][action])

        current_state = new_state
        action = new_action

episode_ended = False
env.reset()
current_state = env.observation
action = np.argmax(Q_table[current_state])
while True:
    env.render()
    obs = env.step(action)
    new_state = obs[0]
    if obs[2]:
        break
    action = np.argmax(Q_table[new_state])
