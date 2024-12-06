import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from qlearning import QLearning

import pygame
from environment import UnknownAngryBirds, PygameInit

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Hidden layer size can be adjusted
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)  # Output layer has 4 values (actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQLearning(QLearning):
    def __init__(self, env, learning_rate=0.5, discount_factor=0.8, epsilon_greedy=0.99, decay_rate=0.99, pigs_num=8):
        super().__init__(env, learning_rate, discount_factor, epsilon_greedy, decay_rate, pigs_num)

        self.input_dim = 10  # i, j and the 8 True/False values
        self.output_dim = 4  # Actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the Q-network
        self.q_network = DQNetwork(self.input_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def predict(self, state, pig_state):
        # Convert to tensor
        input_tensor = torch.tensor(np.concatenate([state, pig_state]), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.q_network(input_tensor).cpu().numpy()

    def update_q_table(self, state, pig_state, action, reward, next_state, next_pig_state, done):
        input_tensor = torch.tensor(np.concatenate([state, pig_state]), dtype=torch.float32).to(self.device)
        target = reward + (1 - done) * self.discount_factor * torch.max(self.q_network(
            torch.tensor(np.concatenate([next_state, next_pig_state]), dtype=torch.float32).to(self.device)))

        # Get current Q-value
        current_q_value = self.q_network(input_tensor)[action]

        # Compute loss
        loss = self.criterion(current_q_value, target)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def episode(self):
        state = self.env.reset()
        pig_state = self.__initial_pig_states
        total_rewards = 0
        done = False

        while not done:
            if np.random.rand() < self.epsilon_greedy:
                action = np.random.choice([0, 1, 2, 3])
            else:
                action = np.argmax(self.predict(state, pig_state))

            next_state, reward, next_pig_state, done = self.env.step(action)
            total_rewards += reward
            self.update_q_table(state, pig_state, action, reward, next_state, next_pig_state, done)

            state, pig_state = next_state, next_pig_state

        return total_rewards

    def set_policy(self):
        # Initialize policy using neural network predictions
        max_indices = np.argmax(self.q_table, axis=3)
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.num_configs):
                    # Use neural network to estimate uninitialized values
                    if np.all(self.q_table[i, j, k, :] == 0):
                        predicted_q_values = self.predict([i, j], self.__initial_pig_states)
                        self.q_table[i, j, k, :] = predicted_q_values
                    max_indices_at_point = np.where(self.q_table[i, j, k, :] == np.max(self.q_table[i, j, k, :]))[0]
                    if len(max_indices_at_point) > 1:
                        max_indices[i, j, k] = np.random.choice(max_indices_at_point)
        self.q_policy = max_indices
        return self.q_policy


if __name__ == "__main__":

    env = UnknownAngryBirds()
    screen, clock = PygameInit.initialization()
    FPS = 10

    ql = DQLearning(env=env, decay_rate=.995, learning_rate=0.9, discount_factor=0.8, epsilon_greedy=0.99)
    values_difference, total_rewards = ql.explore(num_episodes=10000, conv_patience=10, conv_epsilon=10)
    ql.plot_values_difference(values_difference, total_rewards)
    policy = ql.set_policy()
    ql.plot_policy(policy=policy)
    state = env.reset()

    episode_reward = []
    for _ in range(5):

        running = True
        total_reward = 0
        pig_state = [True for _ in range(8)]

        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            env.render(screen)

            action = policy[state[0], state[1], ql.get_config_index(pig_state)]
            next_state, reward, pig_state, done = env.step(action)
            state = next_state
            total_reward += reward

            if done:
                print(pig_state)
                print(f"Episode finished with reward: {total_reward}")
                state = env.reset()
                episode_reward.append(total_reward)
                total_reward = 0
                running = False

            pygame.display.flip()
            clock.tick(FPS)

    print(f'MEAN REWARD: {sum(episode_reward)/len(episode_reward)}')

    pygame.quit()