import numpy as np
from state import State

GAMMA = 0.9
THETA = 22

# THIS CODE IS NOT COMPLETED YET!!!

id_state = {}


def reward(current_state, action):
    new_state = current_state.copy()
    if 0 <= current_state.loc_a + action < 20 and current_state.loc_b - action >= 0:
        new_loc_a = current_state.loc_a + action
        new_loc_b = current_state.loc_b - action
        _id = find_state_id(new_loc_a, new_loc_b)

        new_state = State(new_loc_a, new_loc_b, _id)
    else:
        action = 0

    req_a, req_b, ret_a, ret_b, = np.random.poisson((3, 4, 3, 2))
    if req_a > new_state.loc_a:
        req_a = new_state.loc_a

    if req_b > new_state.loc_b:
        req_b = new_state.loc_b

    while ret_a + ret_b + new_state.loc_a + new_state.loc_b > 20:
        ret_a -= 1
        ret_b -= 1

    if ret_a < 0:
        ret_a = 0
    if ret_b < 0:
        ret_b = 0

    total_reward = -np.abs(action) * 2
    total_reward += 10 * (req_a + req_b)

    new_state.loc_a -= req_a
    new_state.loc_b -= req_b

    if new_state.loc_a + ret_a + new_state.loc_b + ret_b < 20:
        new_state.loc_a += ret_a
        new_state.loc_b += ret_b

    new_state.id = find_state_id(new_state.loc_a, new_state.loc_b)

    return new_state.id, total_reward


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

    return state_value


def policy_improvement(state_value, policy):
    policy_stable = True

    for i in range(len(state_value)):
        state = policy[i]
        old_action = state.best_action()

        best_state_value = -999
        best_action = 0
        for action in range(-5, 6):
            if is_legal_action(state.loc_a, state.loc_b, action):
                if state_value[find_state_id(state.loc_a + action, state.loc_b - action)] > best_state_value:
                    best_state_value = state_value[find_state_id(state.loc_a + action, state.loc_b + action)]
                    best_action = action

        for action_prob in state.action_prob:
            if action_prob[0] != best_action:
                action_prob[1] = 0
            else:
                action_prob[1] = 1

        if old_action != best_action:
            policy_stable = False

    if policy_stable:
        return policy
    else:
        return policy_improvement(policy_evaluation(state_value, policy), policy)


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


def find_state_id(loc_a, loc_b):
    return id_state[(loc_a, loc_b)]


def is_legal_action(loc_a, loc_b, action):
    if 0 <= loc_a + action < 20 and loc_b - action >= 0:
        return True

    return False


def main():
    initial_policy = []
    id_count = 0
    for i in range(21):
        for j in range(21):
            if i + j <= 20:
                initial_policy.append(State(i, j, id_count))
                id_state[(i, j)] = id_count
                id_count += 1

    initial_policy = fix_policy(initial_policy)

    state_value = np.zeros(len(initial_policy))
    state_value = policy_evaluation(state_value, initial_policy)

    policy_improvement(state_value, initial_policy)

    print("the end")


if __name__ == '__main__':
    main()
