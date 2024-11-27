import numpy as np
import matplotlib.pyplot as plt

class QLearning():

    def __init__(self, env, learningRate=0.1, discountFactor=0.8, epsilonGreedy=0.99, decayRate=0.99):
        self.env = env
        self.opt = 4
        self.dim = 8

        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilonGreedy = epsilonGreedy
        self.decayRate = decayRate

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

            next_reward = np.max(self.qTable[next_state[0], next_state[1]]) if not done else 0
            self.qTable[state[0], state[1], action] += self.learningRate * (
                    reward
                    + self.discountFactor * next_reward
                    - self.qTable[state[0], state[1], action]
            )

            state = next_state
            self.epsilonGreedy = max(0.1, self.epsilonGreedy * self.decayRate)
        return total_rewards

    def explore(self, episodes):
        rewards = []
        for episode in range(episodes):
            episode_reward = self.episode()
            rewards.append(episode_reward)

            self.epsilonGreedy = max(0.1, self.epsilonGreedy * self.decayRate)

        return rewards

    def setPolicy(self):
        self.qPolicy = np.argmax(self.qTable, axis=2)
        return self.qPolicy

    # def qTableHeatMap(self):
    #     # Compute max Q-value and best action for each state
    #     max_q_values = np.max(self.qTable, axis=2)  # Max Q-value per state
    #     best_actions = np.argmax(self.qTable, axis=2)  # Best action per state
    #
    #     # Create a heatmap
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(max_q_values, cmap='coolwarm', origin='upper')
    #     plt.colorbar(label="Max Q-value")
    #     plt.title("Q-Table Heatmap with Policy Arrows")
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #
    #     # Overlay arrows for the best policy
    #     for i in range(self.dimension):
    #         for j in range(self.dimension):
    #             action = best_actions[i, j]
    #             if action == 0:  # Up
    #                 plt.arrow(j, i, 0, -0.3, head_width=0.2, head_length=0.2, fc='black', ec='black')
    #             elif action == 1:  # Down
    #                 plt.arrow(j, i, 0, 0.3, head_width=0.2, head_length=0.2, fc='black', ec='black')
    #             elif action == 2:  # Left
    #                 plt.arrow(j, i, -0.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    #             elif action == 3:  # Right
    #                 plt.arrow(j, i, 0.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    #
    #     plt.gca().invert_yaxis()  # Invert y-axis to align with grid indexing
    #     plt.grid(False)
    #     plt.show()
    #
