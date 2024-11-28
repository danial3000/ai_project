import numpy as np
import matplotlib.pyplot as plt


class QLearning:
    action_mapping = {
        0: (0, 1),  # Up
        1: (0, -1),  # Down
        2: (-1, 0),  # Left
        3: (1, 0)  # Right
    }

    def __init__(self, env, learning_rate=0.1, discount_factor=0.8, epsilon_greedy=0.9999, decay_rate=0.99, conv_epsilon=0.001, conv_patience=100000):
        self.env = env
        self.opt = 4
        self.dim = 8

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_greedy = epsilon_greedy
        self.decay_rate = decay_rate

        self.q_table = np.zeros((self.dim, self.dim, self.opt))
        self.q_policy = np.zeros((self.dim, self.dim), dtype=np.int64)


    def episode(self):

        state = self.env.reset()
        total_rewards = 0
        done = False

        state_visit_count = np.full((self.dim, self.dim), dtype=np.int64, fill_value=-1)
        step_count = 0

        while not done:
            step_count += 1

            if np.random.rand() < self.epsilon_greedy:
                action = np.random.choice([0, 1, 2, 3])
            else:
                action = np.argmax(self.q_table[state[0], state[1]])

            next_state, reward, done = self.env.step(action)
            total_rewards += reward

            state_visit_count[state[0], state[1]] += 1
            revisit_penalty = - 0.01 * state_visit_count[state[0], state[1]] * step_count if self.epsilon_greedy <= 0.2 else 0

            next_max_q = np.max(self.q_table[next_state[0], next_state[1]]) if not done else 0
            reward *= (2 - self.epsilon_greedy) if reward < -1000 else 1
            reward += 200 if total_rewards >= 1650 and 0 > reward > -1000 else 0
            self.q_table[state[0], state[1], action] += self.learning_rate * (
                reward
                + self.discount_factor * next_max_q
                - self.q_table[state[0], state[1], action]
                + revisit_penalty
            )

            state = next_state
        return total_rewards

    def explore(self, num_episodes, conv_epsilon, conv_patience):

        q_val_diff_series = []
        total_rewards = []

        conv_count = 0

        for _ in range(num_episodes):
            prev_qt = self.q_table.copy()

            reward = self.episode()
            total_rewards.append(reward)

            value_diff = np.sum(np.abs(self.q_table - prev_qt))
            q_val_diff_series.append(value_diff)

            self.epsilon_greedy = max(0.1, self.epsilon_greedy * self.decay_rate)
            self.learning_rate = max(0.001, self.learning_rate * self.decay_rate)

            if value_diff < conv_epsilon:
                conv_count += 1
            else:
                conv_count = 0
            if conv_count >= conv_patience:
                break

        return q_val_diff_series, total_rewards


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
        self.q_policy = np.argmax(self.q_table, axis=2)
        return self.q_policy


    def plot_policy(self, policy):
        # Set background color
        background_color = (245 / 255, 245 / 255, 220 / 255)  # Normalize RGB values to [0, 1]

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_facecolor(background_color)

        # Set the limits and ticks
        ax.set_xlim(0, self.dim)
        ax.set_ylim(0, self.dim)
        ax.set_xticks(range(self.dim))
        ax.set_yticks(range(self.dim))

        # Draw grid lines
        ax.grid(True, color='black', linestyle='-', linewidth=0.8)

        # Remove axis labels for clarity
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Invert y-axis to have (0,0) in the bottom-left
        plt.gca().invert_yaxis()

        # Plot arrows for each cell
        for i in range(self.dim):
            for j in range(self.dim):
                action = policy[i, j]
                dx, dy = self.action_mapping[action]
                arrow_length = 0.3  # Length of arrows
                dx_norm = dx * arrow_length
                dy_norm = dy * arrow_length
                ax.arrow(j + 0.5, i + 0.5, dx_norm, -dy_norm,
                         head_width=0.1, head_length=0.1, fc='k', ec='k')

        # Set title
        plt.title("Policy Rule after Q-Learning", fontsize=16, color='black')

        # Show the plot
        plt.show()
