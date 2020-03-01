import numpy as np


def reward(current_state, action):
    if current_state == 0 or current_state == 15:
        return current_state, 0
    elif action == 0:
        new_state = current_state - 4 if current_state > 3 else current_state
    elif action == 1:
        new_state = current_state + 1 if current_state != 3 or current_state != 7 or current_state != 11 or current_state != 15 else current_state
    elif action == 2:
        new_state = current_state + 4 if current_state < 12 else current_state
    else:
        new_state = current_state - 1 if current_state != 0 or current_state != 4 or current_state != 8 or current_state != 12 else current_state

    if new_state == 0 or new_state == 15:
        return new_state, 0
    else:
        return new_state, -1


def main():
    actions = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
    gamma = 1
    grid_world = np.zeros(16)
    copy_table = grid_world.copy()

    theta = 0.0001
    delta = 0
    while True:
        for state in range(len(grid_world)):
            grid_world[state] = copy_table[state]

            sum_of_actions = 0
            for a in actions:
                new_state, rew = reward(state, a)
                sum_of_actions += 0.25 * (rew + gamma * copy_table[new_state])

            copy_table[state] = sum_of_actions

            delta = abs(np.sum(grid_world - copy_table))

        if delta < theta:
            break

    print(np.reshape(grid_world, (4, 4)))


if __name__ == '__main__':
    main()
