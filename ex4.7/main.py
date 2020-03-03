import numpy as np
from state import State

GAMMA = 0.9
THETA = 0.0001


# THIS CODE IS NOT COMPLETED YET!!!


# TODO: review the action queue in this function
def reward(current_state, action):
    new_state = State(current_state.loc_a + action, current_state.loc_b - action)

    req_a, req_b, ret_a, ret_b, = np.random.poisson((3, 4, 3, 2))
    if req_a > current_state.loc_a:
        req_a = current_state.loc_a

    if req_b > current_state.loc_b:
        req_b = current_state.loc_b

    while ret_a + ret_b + current_state.loc_a + current_state.loc_b > 20:
        ret_a -= 1
        ret_b -= 1

    total_reward = -np.abs(action) * 2
    total_reward += 10 * (req_a + req_b)

    current_state.loc_a -= req_a
    current_state.loc_b -= req_b

    current_state.loc_a += ret_a
    current_state.loc_b += ret_b

    return new_state, total_reward


def policy_evaluation(state_value, policy):
    copy_value = state_value.copy()

    while True:
        delta = 0
        for i in range(len(state_value)):
            for action_prob in policy[i].action_prob:
                new_state, rew = reward(policy[i], action_prob[0])
                copy_value[i] += action_prob[1] * (rew + GAMMA * state_value[new_state])

            delta = max(delta, np.abs(state_value[i] - copy_value[i]))

        state_value = copy_value.copy()
        copy_value = np.zeros(len(state_value))

        if delta < THETA:
            break

    print(state_value)


def fix_policy(policy):
    for i in range(len(policy)):
        count = 0
        for j in range(11):
            if policy[i].action_prob[j][0] + policy[i].loc_a < 0 or policy[i].action_prob[j][0] + \
                    policy[i].loc_a > 20 or policy[i].action_prob[j][0] > policy[i].loc_b:
                count += 1
                policy[i].action_prob[j][1] = 0

        # maybe more dynamic than just 1 / new_prob
        for j in range(11):
            new_prob = 11 - count
            if policy[i].action_prob[j][1] != 0:
                policy[i].action_prob[j][1] = 1 / new_prob

    return policy


def print_policy(policy):
    for i in policy:
        i.print()


def main():
    initial_policy = []
    for i in range(21):
        for j in range(21):
            if i + j <= 20:
                initial_policy.append(State(i, j))

    initial_policy = fix_policy(initial_policy)

    state_value = np.zeros(len(initial_policy))

    # policy_evaluation(state_value, initial_policy)
    # print_policy(initial_policy)


if __name__ == '__main__':
    main()
