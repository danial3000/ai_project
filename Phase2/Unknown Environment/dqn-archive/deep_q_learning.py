import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from environment import UnknownAngryBirds, PygameInit

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000
TARGET_UPDATE_FREQ = 10

# Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        self.steps_done = 0

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        self.steps_done += 1
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-self.steps_done / EPSILON_DECAY)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Training the Agent
if __name__ == "__main__":
    env = UnknownAngryBirds()
    screen, clock = PygameInit.initialization()
    state_dim = 2  # Grid position (row, col)
    action_dim = 4  # Actions: Up, Down, Left, Right

    agent = DQNAgent(state_dim, action_dim)
    episodes = 500
    FPS = 30

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, _, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            env.render(screen)
            pygame.display.flip()
            clock.tick(FPS)

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    pygame.quit()
