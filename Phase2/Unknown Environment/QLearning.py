import numpy as np
import matplotlib.pyplot as plt

class QLearning:
    action_mapping = {
        0: (0, 1),  # Up
        1: (0, -1),  # Down
        2: (-1, 0),  # Left
        3: (1, 0)  # Right
    }

    def __init__(self, env, learning_rate=0.1, discount_factor=0.8, epsilon_greedy=0.99, decay_rate=0.99):
        self.env = env
        self.opt = 4
        self.dim = 8

        self.learningRate = learning_rate
        self.discountFactor = discount_factor
        self.epsilonGreedy = epsilon_greedy
        self.decayRate = decay_rate

        self.qTable = np.zeros((self.dim, self.dim, self.opt))
        self.qPolicy = np.zeros((self.dim, self.dim), dtype=np.int64)


    def episode(self):
        state = self.env.reset()
        total_rewards = 0
        done = False

        while not done:
            if np.random.rand() <= self.epsilonGreedy:
                action = np.random.choice([0, 1, 2, 3])
            else:
                action = np.argmax(self.qTable[state])

            next_state, reward, done = self.env.step(action)
            total_rewards += reward

            next_reward = np.max(self.qTable[next_state]) if not done else 0
            self.qTable[state, action] += self.learningRate * (
                    reward
                    + self.discountFactor * next_reward
                    - self.qTable[state, action]
            )

            state = next_state
            self.epsilonGreedy = max(0.1, self.epsilonGreedy * self.decayRate)
        return total_rewards


    def explore(self, episodes):
        values_difference = []
        total_rewards = []

        for episode in range(episodes):
            prev_q = self.qTable.copy()

            episode_reward = self.episode()

            total_rewards.append(episode_reward)
            values_difference.append(np.sum(np.abs(prev_q - episode_reward)))
            self.epsilonGreedy = max(0.1, self.epsilonGreedy * self.decayRate)

        return values_difference, total_rewards

    @staticmethod
    def plot_values_difference(values_diff, episodes):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Q-Table differences on the left y-axis
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Mean Absolute Difference', color='blue')
        ax1.plot(range(1, len(values_diff) + 1), values_diff, label='Q-Table Difference', marker='.', linestyle='None',
                 color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Reward', color='red')
        ax2.plot(range(1, len(episodes) + 1), episodes, label='Episode Reward', marker='.', linestyle='None',
                 color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Add title and legends
        fig.suptitle('Q-Table Value Difference and Reward per Episode')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()


    def set_policy(self):
        self.qPolicy = np.argmax(self.qTable, axis=2)
        return self.qPolicy

    def plot_policy(self, policy):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Set the limits and ticks
        ax.set_xlim(-0.5, self.dim - 0.5)
        ax.set_ylim(-0.5, self.dim - 0.5)
        ax.set_xticks(range(self.dim))
        ax.set_yticks(range(self.dim))

        # Draw grid lines
        ax.grid(True)

        # Remove axis labels for clarity
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Invert y-axis to have (0,0) at the top-left corner
        plt.gca().invert_yaxis()

        # Plot arrows for each cell
        for i in range(self.dim):
            for j in range(self.dim):
                action = policy[i, j]
                dx, dy = self.action_mapping[action]
                arrow_length = 0.3
                dx_norm = dx * arrow_length
                dy_norm = dy * arrow_length
                ax.arrow(j, i, dx_norm, -dy_norm, head_width=0.1, head_length=0.1, fc='k', ec='k')


        plt.title("Policy Rule after Q-Learning")
        plt.show()

