import numpy as np

ALPHA = 0.1
STEP_SIZE = 200000


def main():
    rewards = np.empty([10, 10])
    means = np.array([0.2, -0.9, 1.55, 0.45, 1.2, -1.3, -0.2, -1, 0.8, -0.6])

    for i in range(10):
        rewards[i] = np.random.normal(means[i], 1, 10)

    original_rewards = rewards.copy()

    # epsilon-greedy
    EPSILON = 1 / 128

    for _ in range(5):
        Q_table = np.zeros(10)
        EPSILON *= 2
        for i in range(STEP_SIZE):
            selected_bandit = np.argmax(Q_table) if np.random.random() > EPSILON else np.random.randint(10)

            rw = np.random.choice(rewards[selected_bandit])
            rewards += np.random.normal(0, 0.01, (10, 10))

            Q_table[selected_bandit] = Q_table[selected_bandit] + ALPHA * (
                    rw - Q_table[selected_bandit])

    # optimistic initialization
    rewards = original_rewards
    Q_table = np.ones(10) * 4
    n = 0

    for i in range(STEP_SIZE):
        selected_bandit = np.argmax(Q_table)

        rw = np.random.choice(rewards[selected_bandit])
        rewards += np.random.normal(0, 0.01, (10, 10))
        n += 1

        Q_table[selected_bandit] = Q_table[selected_bandit] + 1 / n * (
                rw - Q_table[selected_bandit])

    # UCB
    c = 1 / 16
    for _ in range(6):
        c *= 2

        rewards = original_rewards
        Q_table = np.ones(10) * 4
        n = 0
        selected_numbers = np.ones(10)

        for t in range(STEP_SIZE):
            selected_bandit = np.argmax(Q_table + c * np.sqrt(np.log(t + 1) / selected_numbers))
            selected_numbers[selected_bandit] += 1

            rw = np.random.choice(rewards[selected_bandit])
            rewards += np.random.normal(0, 0.01, (10, 10))
            n += 1

            Q_table[selected_bandit] = Q_table[selected_bandit] + 1 / n * (
                    rw - Q_table[selected_bandit])


if __name__ == '__main__':
    main()
