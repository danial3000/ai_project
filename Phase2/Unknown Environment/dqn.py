import  torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

import pygame
from environment import UnknownAngryBirds, PygameInit

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # لایه پنهان اول با 128 نورون
        self.fc2 = nn.Linear(128, 64)         # لایه پنهان دوم با 64 نورون
        self.fc3 = nn.Linear(64, output_dim)  # لایه خروجی با تعداد نورون برابر تعداد اقدامات

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # اعمال تابع فعال‌سازی ReLU پس از لایه اول
        x = torch.relu(self.fc2(x))  # اعمال تابع فعال‌سازی ReLU پس از لایه دوم
        return self.fc3(x)            # خروجی Q-values برای هر اقدام

class DQLearning:
    def __init__(self, env, learning_rate=1e-1, discount_factor=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay_steps=5000, batch_size=64, memory_size=100000,
                 target_update_freq=1000):

        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_step = (epsilon_start - epsilon_end) / epsilon_decay_steps

        self.input_dim = 64  # 8x8 grid
        self.output_dim = 4  # Actions: Up, Down, Left, Right

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the Q-network and target network
        self.q_network = DQNetwork(self.input_dim, self.output_dim).to(self.device)
        self.target_network = DQNetwork(self.input_dim, self.output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        # For updating target network
        self.target_update_freq = target_update_freq
        self.step_count = 0

    def get_state_vector(self, grid, agent_position):

        state_vector = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i, j) == agent_position:
                    state_vector.append(500)  # مقدار نمایانگر موقعیت ایجنت
                else:
                    cell = grid[i][j]
                    if cell == 'R':         # دیوار
                        state_vector.append(-50)
                    elif cell == 'P':       # پگ
                        state_vector.append(250)
                    elif cell == 'Q':       # ملکه
                        state_vector.append(-400)
                    elif cell == 'G':       # هدف (تخم مرغ)
                        state_vector.append(2000)
                    elif cell == 'TNT':     # TNT
                        state_vector.append(-3500)
                    else:                   # خانه خالی یا محتوای حذف شده
                        state_vector.append(0)
        return np.array(state_vector, dtype=np.float32)

    def select_action(self, state_vector):

        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)  # Explore: random action
        else:
            state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # Exploit: best action

    def store_transition(self, state_vector, action, reward, next_state_vector, done):

        self.memory.append((state_vector, action, reward, next_state_vector, done))

    def sample_memory(self):

        return random.sample(self.memory, self.batch_size)

    def update_epsilon(self):

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        else:
            self.epsilon = self.epsilon_end

    def update_q_network(self):

        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train

        mini_batch = self.sample_memory()
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert lists of arrays to single NumPy arrays for efficiency
        states = np.array(states)
        next_states = np.array(next_states)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions)

        # Compute target Q values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.discount_factor * max_next_q_values * (1 - dones)

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.update_epsilon()

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train_episode(self):

        agent_pos = self.env.reset()  # دریافت موقعیت ایجنت
        grid = self.env._UnknownAngryBirds__grid  # دسترسی به شبکه از طریق name mangling
        state_vector = self.get_state_vector(grid, agent_pos)
        total_reward = 0
        done = False

        while not done:
            action = self.select_action(state_vector)
            next_state, reward, next_pig_state, done = self.env.step(action)
            grid = self.env._UnknownAngryBirds__grid  # بروزرسانی شبکه پس از اقدام
            next_agent_pos = self.env._UnknownAngryBirds__agent_pos  # دریافت موقعیت جدید ایجنت
            next_state_vector = self.get_state_vector(grid, next_agent_pos)

            self.store_transition(state_vector, action, reward, next_state_vector, done)
            self.update_q_network()

            state_vector = next_state_vector
            total_reward += reward

        return total_reward

    def explore(self, num_episodes, conv_patience=100, conv_epsilon=1e-3):

        rewards_per_episode = []
        recent_rewards = deque(maxlen=conv_patience)
        avg_rewards = []

        for episode in range(1, num_episodes + 1):
            reward = self.train_episode()
            rewards_per_episode.append(reward)
            recent_rewards.append(reward)

            if episode % 100 == 0:
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_rewards.append(avg_reward)
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")

                # Check for convergence
                if len(recent_rewards) == conv_patience and np.std(recent_rewards) < conv_epsilon:
                    print(f"Converged after {episode} episodes.")
                    break

        return rewards_per_episode, avg_rewards

    def set_policy(self):

        policy = {}
        grid_size = self.env._UnknownAngryBirds__grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                # ایجاد یک شبکه با ایجنت در موقعیت (i, j) و سایر سلول‌ها خالی
                grid = [['T' for _ in range(grid_size)] for _ in range(grid_size)]
                grid[i][j] = 'Agent'  # اضافه کردن ایجنت
                state_vector = self.get_state_vector(grid, (i, j))
                state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()
                policy[(i, j)] = action
        return policy

    def plot_values_difference(self, rewards, avg_rewards):

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(rewards, label='Rewards per Episode', alpha=0.6)
        plt.plot(range(0, len(rewards), 100), avg_rewards, label='Average Rewards (100 episodes)', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards over Episodes')
        plt.legend()
        plt.show()

    def plot_policy(self, policy):

        import matplotlib.pyplot as plt

        grid_size = self.env._UnknownAngryBirds__grid_size
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks(range(grid_size + 1))
        ax.set_yticks(range(grid_size + 1))
        ax.grid(True)

        # Invert y-axis to match grid coordinates
        plt.gca().invert_yaxis()

        # Initialize action_map as a list of lists to hold actions for each cell
        action_map = [[[] for _ in range(grid_size)] for _ in range(grid_size)]

        # Populate action_map with actions from the policy
        for (i, j), action in policy.items():
            if 0 <= i < grid_size and 0 <= j < grid_size:
                action_map[i][j].append(action)

        # Determine the most common action for each cell and plot arrows
        for i in range(grid_size):
            for j in range(grid_size):
                actions = action_map[i][j]
                if actions:
                    # Compute the most common action (mode) for the cell
                    action = max(set(actions), key=actions.count)
                    dx, dy = self.get_action_direction(action)
                    ax.arrow(
                        j + 0.5, i + 0.5,  # Starting point (center of the cell)
                        dx * 0.3, dy * 0.3,  # Direction and length of the arrow
                        head_width=0.1, head_length=0.1,
                        fc='k', ec='k'
                    )

        plt.title("Learned Policy Arrows")
        plt.show()

    @staticmethod
    def get_action_direction(action):

        action_mapping = {
            0: (0, -1),  # Up
            1: (0, 1),   # Down
            2: (-1, 0),  # Left
            3: (1, 0)    # Right
        }
        return action_mapping.get(action, (0, 0))

    def save_model(self, filepath):

        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):

        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Initialize the environment and Pygame
    env = UnknownAngryBirds()
    screen, clock = PygameInit.initialization()
    FPS = 10

    # Initialize DQN agent
    dq_agent = DQLearning(
        env=env,
        learning_rate=1e-3,
        discount_factor=0.5,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=5000,
        batch_size=64,
        memory_size=100000,
        target_update_freq=1000
    )

    # Train the agent
    num_training_episodes = 5000
    rewards, avg_rewards = dq_agent.explore(num_episodes=num_training_episodes, conv_patience=100, conv_epsilon=1e-3)
    dq_agent.plot_values_difference(rewards, avg_rewards)

    # Extract and visualize the policy
    print("Extracting policy...")
    policy = dq_agent.set_policy()
    print("Policy extraction complete. Visualizing policy...")
    dq_agent.plot_policy(policy=policy)

    # Demonstrate the learned policy
    print("Demonstrating learned policy...")
    num_demo_episodes = 5
    episode_rewards = []
    for demo_episode in range(1, num_demo_episodes + 1):
        agent_pos = env.reset()
        grid = env._UnknownAngryBirds__grid
        state_vector = dq_agent.get_state_vector(grid, agent_pos)
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            env.render(screen)

            # Select action based on the trained Q-network
            action = dq_agent.select_action(state_vector)

            # Execute action
            next_state, reward, next_pig_state, done = env.step(action)
            grid = env._UnknownAngryBirds__grid
            next_agent_pos = env._UnknownAngryBirds__agent_pos
            next_state_vector = dq_agent.get_state_vector(grid, next_agent_pos)
            total_reward += reward

            # Update state
            agent_pos = next_agent_pos
            state_vector = next_state_vector

            pygame.display.flip()
            clock.tick(FPS)

        print(f"Demo Episode {demo_episode} finished with reward: {total_reward}")
        episode_rewards.append(total_reward)

    # Calculate and display mean reward
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    print(f'MEAN REWARD over {num_demo_episodes} episodes: {mean_reward}')
    pygame.quit()
