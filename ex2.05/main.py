import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.1
ALPHA = 0.1

TIMES = 5000
STEP_SIZE = 1000


def main():
    plt.xlabel('Steps')
    plt.ylabel('Average reward')

    rewards = np.empty([10, 10])
    means = np.array([0.2, -0.9, 1.55, 0.45, 1.2, -1.3, -0.2, -1, 0.8, -0.6])

    for i in range(10):
        rewards[i] = np.random.normal(means[i], 1, 10)

    original_rewards = rewards.copy()

    average_rewards1 = np.empty(STEP_SIZE)
    average_rewards2 = np.empty(STEP_SIZE)

    # 1/n step size version
    for _ in range(TIMES):
        first_Q_table = np.zeros(10)
        n = 0
        rewards = original_rewards

        print(f'{_} / {TIMES} (1)')

        for i in range(STEP_SIZE):
            action = np.argmax(first_Q_table) if np.random.random() > EPSILON else np.random.randint(10)

            rw = np.random.choice(rewards[action])
            average_rewards1[i] += rw
            n += 1

            rewards += np.random.normal(0, 0.01, (10, 10))

            first_Q_table[action] = first_Q_table[action] + 1 / n * (
                    rw - first_Q_table[action])

    average_rewards1 /= TIMES

    # Constant step size version
    for _ in range(TIMES):
        second_Q_table = np.zeros(10)
        rewards = original_rewards

        print(f'{_} / {TIMES} (2)')

        for i in range(STEP_SIZE):
            action = np.argmax(second_Q_table) if np.random.random() > EPSILON else np.random.randint(10)

            rw = np.random.choice(rewards[action])
            average_rewards2[i] += rw

            rewards += np.random.normal(0, 0.01, (10, 10))

            second_Q_table[action] = second_Q_table[action] + ALPHA * (
                    rw - second_Q_table[action])

    average_rewards2 /= TIMES

    plt.plot(average_rewards1, 'g')
    plt.plot(average_rewards2, 'r')
    plt.show()


if __name__ == '__main__':
    main()
