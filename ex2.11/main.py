import numpy as np

ALPHA = 0.1
STEP_SIZE = 200000


def main():
    rewards = np.empty([10, 10])
    means = np.array([0.2, -0.9, 1.55, 0.45, 1.2, -1.3, -0.2, -1, 0.8, -0.6])
    mean_rewards = {'epsilon-greedy': np.empty(6), 'optimistic-init': np.empty(5), 'ucb': np.empty(7)}

    for i in range(10):
        rewards[i] = np.random.normal(means[i], 1, 10)

    original_rewards = rewards.copy()

    # epsilon-greedy
    EPSILON = 1 / 128
    for j in range(6):
        Q_table = np.zeros(10)
        rewards = original_rewards

        for i in range(STEP_SIZE):
            selected_bandit = np.argmax(Q_table) if np.random.random() > EPSILON else np.random.randint(10)

            rw = np.random.choice(rewards[selected_bandit])
            if i > 100000:
                mean_rewards['epsilon-greedy'][j] += rw

            rewards += np.random.normal(0, 0.01, (10, 10))

            Q_table[selected_bandit] = Q_table[selected_bandit] + ALPHA * (
                    rw - Q_table[selected_bandit])

        EPSILON *= 2

    print("epsilon-greedy finished")

    # optimistic initialization
    for j in range(5):
        Q_table = np.ones(10) * 1 / 4 * (2 ** j)
        rewards = original_rewards
        n = 0

        for i in range(STEP_SIZE):
            selected_bandit = np.argmax(Q_table)

            rw = np.random.choice(rewards[selected_bandit])
            if i > 100000:
                mean_rewards['optimistic-init'][j] += rw

            rewards += np.random.normal(0, 0.01, (10, 10))
            n += 1

            Q_table[selected_bandit] = Q_table[selected_bandit] + 1 / n * (
                    rw - Q_table[selected_bandit])

    print("optimistic initialization finished")

    # UCB
    c = 1 / 16
    for j in range(7):
        rewards = original_rewards
        Q_table = np.ones(10) * 4
        n = 0
        selected_numbers = np.ones(10)

        for t in range(STEP_SIZE):
            selected_bandit = np.argmax(Q_table + c * np.sqrt(np.log(t + 1) / selected_numbers))
            selected_numbers[selected_bandit] += 1

            rw = np.random.choice(rewards[selected_bandit])
            if t > 100000:
                mean_rewards['ucb'][j] += rw

            rewards += np.random.normal(0, 0.01, (10, 10))
            n += 1

            Q_table[selected_bandit] = Q_table[selected_bandit] + 1 / n * (
                    rw - Q_table[selected_bandit])

        c *= 2

    print("ucb finished\n")

    # Take means of dictionary
    for key in mean_rewards:
        for i in range(len(mean_rewards[key])):
            mean_rewards[key][i] /= (STEP_SIZE / 2)

    print(mean_rewards)


if __name__ == '__main__':
    main()
