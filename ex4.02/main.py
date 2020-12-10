import numpy as np


def reward(current_state, action):
    if current_state == 0 or current_state == 15:
        return current_state, 0
    elif current_state == 16:
        if action == 0:
            new_state = 13
        elif action == 1:
            new_state = 14
        elif action == 2:
            new_state = 16
        else:
            new_state = 12
    elif action == 0:
        new_state = current_state - 4 if current_state > 3 else current_state
    elif action == 1:
        new_state = current_state + 1 if current_state != 3 and current_state != 7 and current_state != 11 and current_state != 15 else current_state
    elif action == 2:
        new_state = current_state + 4 if current_state < 12 else current_state
    else:
        new_state = current_state - 1 if current_state != 0 and current_state != 4 and current_state != 8 and current_state != 12 else current_state

    return new_state, -1


def main():
    actions = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
    discount_factor = 1

    grid_world = np.zeros(17)
    copy_world = grid_world.copy()

    theta = 0.00001
    while True:
        delta = 0

        for state in range(len(grid_world)):
            for a in actions:
                new_state, rew = reward(state, a)
                copy_world[state] += 0.25 * (rew + discount_factor * grid_world[new_state])

            delta = max(delta, np.abs(grid_world[state] - copy_world[state]))

        grid_world = copy_world.copy()
        copy_world = np.zeros(17)

        if delta < theta:
            break

    print(np.reshape(grid_world[:-1], (4, 4)))
    print(f'\t\t\t   {grid_world[-1]}')


if __name__ == '__main__':
    main()
