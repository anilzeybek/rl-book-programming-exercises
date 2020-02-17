import numpy as np

EPSILON = 0.1
ALPHA = 0.1


def main():
    rewards = np.empty([10, 10])
    means = np.array([0.2, -0.9, 1.55, 0.45, 1.2, -1.3, -0.2, -1, 0.8, -0.6])

    for i in range(10):
        rewards[i] = np.random.normal(means[i], 1, 10)

    original_rewards = rewards.copy()

    first_Q_table = np.zeros(10)
    second_Q_table = np.zeros(10)

    n = 0

    first_total_reward = 0
    for i in range(100000):
        selected_bandit = np.argmax(first_Q_table) if np.random.random() > EPSILON else np.random.randint(10)

        rw = np.random.choice(rewards[selected_bandit])
        first_total_reward += rw
        n += 1

        rewards += np.random.normal(0, 0.01, (10, 10))

        first_Q_table[selected_bandit] = first_Q_table[selected_bandit] + 1 / n * (
                rw - first_Q_table[selected_bandit])

    rewards = original_rewards
    second_total_reward = 0
    for i in range(100000):
        selected_bandit = np.argmax(second_Q_table) if np.random.random() > EPSILON else np.random.randint(10)

        rw = np.random.choice(rewards[selected_bandit])
        second_total_reward += rw

        rewards += np.random.normal(0, 0.01, (10, 10))

        second_Q_table[selected_bandit] = second_Q_table[selected_bandit] + ALPHA * (
                rw - second_Q_table[selected_bandit])

    print(f'first: {first_total_reward}, second: {second_total_reward}')


if __name__ == '__main__':
    main()
