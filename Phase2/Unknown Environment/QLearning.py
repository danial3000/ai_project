import numpy as np
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


class QLearning:
    __action_mapping = {
        0: (0, 1),  # Up
        1: (0, -1),  # Down
        2: (-1, 0),  # Left
        3: (1, 0)  # Right
    }


    def __init__(self, env, learning_rate=0.1, discount_factor=0.8, epsilon_greedy=0.9999, decay_rate=0.999, pigs_num=8):
        self.env = env
        self.opt = 4
        self.dim = 8

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_greedy = epsilon_greedy
        self.decay_rate = decay_rate

        self.num_configs = 2 ** pigs_num

        self.__initial_pig_states = [True for _ in range(pigs_num)]

        self.q_table = np.zeros((self.dim, self.dim, self.num_configs, self.opt))
        self.q_policy = np.zeros((self.dim, self.dim, self.num_configs), dtype=np.int64)

    @staticmethod
    def get_config_index(pig_states):
        index = 0
        for i, state in enumerate(pig_states):
            if state:
                index += 2 ** i
        return index

    def episode(self):

        state = self.env.reset()
        config_index = self.get_config_index(self.__initial_pig_states)

        total_rewards = 0
        done = False

        state_visit_count = np.full((self.dim, self.dim), dtype=np.int64, fill_value=-1)
        step_count = 0

        while not done:
            step_count += 1

            if np.random.rand() < self.epsilon_greedy:
                action = np.random.choice([0, 1, 2, 3])
            else:
                action = np.argmax(self.q_table[state[0], state[1], config_index])

            next_state, reward, next_pig_state, done = self.env.step(action)
            total_rewards += reward
            next_config_index = self.get_config_index(next_pig_state)

            state_visit_count[state[0], state[1]] += 1

            if reward == -2000:
                self.q_table[state[0], state[1], :, action] = (np.max(self.q_table[state[0], state[1], :, action])
                                                               + self.learning_rate * (
                                                                       reward
                                                                       - np.max(self.q_table[state[0], state[1], :, action])
                                                               ))
            else:

                next_max_q = np.max(self.q_table[next_state[0], next_state[1], next_config_index]) if not done else 0
                # revisit_penalty = - (1 - self.epsilon_greedy) ** 4 * state_visit_count[state[0], state[1]] * step_count
                reward = -1 if reward == -1000 else reward
                # reward *= 2 - self.epsilon_greedy if reward == -2000 else 1
                # reward = -200 if total_rewards >= 2000 and reward == -400 else reward
                self.q_table[state[0], state[1], config_index, action] += self.learning_rate * (
                        reward
                        + self.discount_factor * next_max_q
                        - self.q_table[state[0], state[1], config_index, action]
                        # + revisit_penalty
                )

            config_index = next_config_index
            state = next_state
        return total_rewards

    def explore(self, num_episodes, conv_epsilon, conv_patience):

        q_val_diff_series = [0]
        total_rewards = [-4000]

        conv_count = 0

        for r in range(num_episodes):
            prev_qt = self.q_table.copy()

            reward = self.episode()
            total_rewards.append(reward)

            value_diff = np.sum(np.abs(self.q_table - prev_qt))
            q_val_diff_series.append(value_diff)

            if total_rewards[-1] <= total_rewards[-2]:
                self.epsilon_greedy *= self.decay_rate

            if q_val_diff_series[-1] >= q_val_diff_series[-2]:
                self.learning_rate *= self.decay_rate


            if value_diff < conv_epsilon:
                conv_count += 1
            else:
                conv_count = 0
            if conv_count >= conv_patience:
                break

        # self.normalize_qtable()

        return q_val_diff_series, total_rewards


    # def normalize_qtable(self):
    #     min_values = np.min(self.q_table, axis=2, keepdims=True)
    #     max_values = np.max(self.q_table, axis=2, keepdims=True)
    #     self.q_table = (self.q_table - min_values) / (max_values - min_values + 1e-8)

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


    @staticmethod
    def triangulation_for_triheatmap(M, N):
            xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the squares
            xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the squares
            x = np.concatenate([xv.ravel(), xc.ravel()])
            y = np.concatenate([yv.ravel(), yc.ravel()])
            cstart = (M + 1) * (N + 1)  # indices of the centers

            # Define triangles for each direction (N, E, S, W)
            trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M) for j in range(N) for i in
                          range(M)]
            trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M) for j in range(N) for i
                          in range(M)]
            trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M) for j in range(N) for i
                          in range(M)]
            trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M) for j in range(N) for i in
                          range(M)]

            return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]

    def plot_qtable_heatmap(self):
        actions_u = self.q_table[:, :, 0]  # Values for up
        actions_r = self.q_table[:, :, 3]  # Values for right
        actions_d = self.q_table[:, :, 1]  # Values for down
        actions_l = self.q_table[:, :, 2]  # Values for West left
        action_values = [actions_u, actions_r, actions_d, actions_l]
        triangul = self.triangulation_for_triheatmap(self.dim, self.dim)

        # Set color maps to range from red to green
        cmaps = ['RdYlGn', 'RdYlGn', 'RdYlGn', 'RdYlGn']
        norms = [plt.Normalize(0, 1) for _ in range(4)]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot each direction with its corresponding color map
        imgs = [ax.tripcolor(t, np.ravel(val), cmap=cmap, norm=norm, ec='white')
                for t, val, cmap, norm in zip(triangul, action_values, cmaps, norms)]

        # Adjust the axis and grid for better visualization
        ax.set_xticks(range(self.dim))
        ax.set_yticks(range(self.dim))
        ax.invert_yaxis()
        ax.set_aspect('equal', 'box')  # Equal aspect ratio to ensure square cells
        plt.tight_layout()

        # Add colorbar for the North direction (since the colormap is shared)
        fig.colorbar(imgs[0], ax=ax)

        # Show the plot
        plt.show()

    def set_policy(self):
        self.q_policy = np.argmax(self.q_table, axis=3)
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
                dx, dy = self.__action_mapping[action]
                arrow_length = 0.3  # Length of arrows
                dx_norm = dx * arrow_length
                dy_norm = dy * arrow_length
                ax.arrow(j + 0.5, i + 0.5, dx_norm, -dy_norm,
                         head_width=0.1, head_length=0.1, fc='k', ec='k')

        # Set title
        plt.title("Policy Rule after Q-Learning", fontsize=16, color='black')

        # Show the plot
        plt.show()
