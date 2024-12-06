"""
Project Title: MDP

Department:
    Artificial Intelligence Department
    Faculty of Computer Engineering
    University of Isfahan
    November 1, 2024

Supervisor:
    Dr. Hossein Karshenas(h.karshenas@eng.ui.ac.ir) - Professor
    Pouria Sameti(pouria.sameti2002@mehr.ui.ac.ir) - Teaching Assistant


Project Overview:
    This project provides a flexible framework within the MDP (Markov Decision Process) setting.
    Students are encouraged to implement various algorithms, such as value iteration, policy
    iteration, and Q-learning, to explore different approaches to decision-making, rewards,
    penalties, and policy evaluation in a stochastic environment.

Objectives:
    - To provide students with practical experience in implementing reinforcement learning algorithms in a stochastic environment.
    - To explore modifications in the MDP framework that introduce intermediate states, rewarding or penalizing agents based on their choices.
    - To challenge students to strategize their approach in reaching the final state while meeting a cumulative score requirement.

Licensing Information:
    -You are free to use or extend these projects for educational purposes.
"""

import numpy as np
import pygame
import random
import copy
import math
from sklearn.metrics import silhouette_score
import itertools

#######################################################
#                DONT CHANGE THIS PART                #
#######################################################
COLORS = {
    'T': (135, 206, 235),  # Tile ground
    'P': (135, 206, 235),  # Pigs
    'Q': (135, 206, 235),  # Queen
    'G': (135, 206, 235),  # Goal
    'R': (135, 206, 235),  # Rock
}

GOOD_PIG_REWARD = 250
GOAL_REWARD = 400
QUEEN_REWARD = -400
DEFAULT_REWARD = (-1)

PIGS = 8
QUEENS = 2
ROCKS = 8


#######################################################
#                DONT CHANGE THIS PART                #
#######################################################


class PygameInit:

    @classmethod
    def initialization(cls):
        grid_size = 8
        tile_size = 100

        pygame.init()
        screen = pygame.display.set_mode((grid_size * tile_size, grid_size * tile_size))
        pygame.display.set_caption("MDP Angry Birds")
        clock = pygame.time.Clock()

        return screen, clock


#######################################################
#                DONT CHANGE THIS PART                #
#######################################################
class AngryBirds:
    def __init__(self):
        self.__grid_size = 8
        self.__tile_size = 100
        self.__num_pigs = PIGS
        self.__num_queens = QUEENS
        self.__num_rocks = ROCKS
        self.__probability_dict = self.__generate_probability_dict()
        self.__base_grid = self.__generate_grid()
        self.__agent_pos = (0, 0)

        self.grid = copy.deepcopy(self.__base_grid)
        self.reward = 0
        self.done = False
        # Add it myself
        self.pig_points_for_cluster = self.get_pig_points()
        self.cluster_points, self.list_of_clusters_points, self.centroid_to_cluster_dict = self.best_clustering(
            self.pig_points_for_cluster)
        self.reward_map = self.reward_function()
        self.transition_table = self.__calculate_transition_model(self.__grid_size, self.__probability_dict,
                                                                  self.reward_map)

        self.__agent_image = pygame.image.load("Env/icons/angry-birds.png")
        self.__agent_image = pygame.transform.scale(self.__agent_image, (self.__tile_size, self.__tile_size))

        self.__pig_image = pygame.image.load('Env/icons/pigs.png')
        self.__pig_image = pygame.transform.scale(self.__pig_image, (self.__tile_size, self.__tile_size))
        self.__pig_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__pig_with_background.fill((135, 206, 235))
        self.__pig_with_background.blit(self.__pig_image, (0, 0))

        self.__egg_image = pygame.image.load('Env/icons/eggs.png')
        self.__egg_image = pygame.transform.scale(self.__egg_image, (self.__tile_size, self.__tile_size))
        self.__egg_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__egg_with_background.fill((135, 206, 235))
        self.__egg_with_background.blit(self.__egg_image, (0, 0))

        self.__queen_image = pygame.image.load('Env/icons/queen.png')
        self.__queen_image = pygame.transform.scale(self.__queen_image, (self.__tile_size, self.__tile_size))
        self.__queen_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__queen_with_background.fill((135, 206, 235))
        self.__queen_with_background.blit(self.__queen_image, (0, 0))

        self.__rock_image = pygame.image.load('Env/icons/rocks.png')
        self.__rock_image = pygame.transform.scale(self.__rock_image, (self.__tile_size, self.__tile_size))
        self.__rock_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__rock_with_background.fill((135, 206, 235))
        self.__rock_with_background.blit(self.__rock_image, (0, 0))

        # add it myself
        # self.action_taken = self.actions_model()

    def get_pig_points(self):
        points_list = list()
        for row in range(self.__grid_size):
            for col in range(self.__grid_size):
                if self.grid[row][col] == 'P':
                    points_list.append((row, col))
        return points_list

    def __generate_grid(self):

        while True:
            filled_spaces = [(0, 0), (self.__grid_size - 1, self.__grid_size - 1)]
            grid = [['T' for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]

            # grid[4][5] = 'Q'
            # grid[0][2] = 'Q'
            #
            # grid[3][5] = 'R'
            # grid[4][2] = 'R'
            # grid[5][1] = 'R'
            # grid[6][0] = 'R'
            # grid[7][0] = 'R'
            # grid[7][4] = 'R'
            # grid[6][6] = 'R'
            # grid[6][5] = 'R'
            #
            # grid[0][6] = 'P'
            # grid[0][4] = 'P'
            # grid[2][5] = 'P'
            # grid[3][7] = 'P'
            # grid[4][0] = 'P'
            # grid[4][6] = 'P'
            # grid[1][2] = 'P'
            # grid[0][1] = 'P'
            num_pigs = self.__num_pigs
            for _ in range(num_pigs):
                while True:
                    r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                    if (r, c) not in filled_spaces:
                        grid[r][c] = 'P'
                        filled_spaces.append((r, c))
                        break

            for _ in range(self.__num_queens):
                while True:
                    r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                    if (r, c) not in filled_spaces:
                        grid[r][c] = 'Q'
                        filled_spaces.append((r, c))
                        break

            for _ in range(self.__num_rocks):
                while True:
                    r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                    if (r, c) not in filled_spaces:
                        grid[r][c] = 'R'
                        filled_spaces.append((r, c))
                        break

            grid[self.__grid_size - 1][self.__grid_size - 1] = 'G'
            if AngryBirds.__is_path_exists(grid=grid, start=(0, 0), goal=(7, 7)):
                break

        return grid

    def reset(self):
        self.grid = copy.deepcopy(self.__base_grid)
        self.__agent_pos = (0, 0)
        self.reward = 0
        self.done = False
        return self.__agent_pos

    def step(self, action):
        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1)  # Right
        }

        neighbors = {
            0: [2, 3],
            1: [2, 3],
            2: [0, 1],
            3: [0, 1]
        }

        intended_probability = self.__probability_dict[self.__agent_pos][action]['intended']
        neighbors_probability = self.__probability_dict[self.__agent_pos][action]['neighbor']

        prob_dist = [0, 0, 0, 0]
        prob_dist[action] = intended_probability
        prob_dist[neighbors[action][0]] = neighbors_probability
        prob_dist[neighbors[action][1]] = neighbors_probability

        chosen_action = np.random.choice([0, 1, 2, 3], p=prob_dist)

        dx, dy = actions[chosen_action]
        new_row = self.__agent_pos[0] + dx
        new_col = self.__agent_pos[1] + dy

        if (0 <= new_row < self.__grid_size and 0 <= new_col < self.__grid_size and
                self.grid[new_row][new_col] != 'R'):
            self.__agent_pos = (new_row, new_col)

        current_tile = self.grid[self.__agent_pos[0]][self.__agent_pos[1]]
        reward = DEFAULT_REWARD

        if current_tile == 'Q':
            reward = QUEEN_REWARD
            self.grid[self.__agent_pos[0]][self.__agent_pos[1]] = 'T'

        elif current_tile == 'P':
            reward = GOOD_PIG_REWARD
            self.grid[self.__agent_pos[0]][self.__agent_pos[1]] = 'T'

        elif current_tile == 'G':
            reward = GOAL_REWARD
            self.done = True

        elif current_tile == 'T':
            reward = DEFAULT_REWARD

        probability = prob_dist[chosen_action]
        self.reward = reward
        next_state = self.__agent_pos
        is_terminated = self.done
        return next_state, probability, self.reward, is_terminated

    def render(self, screen):
        for r in range(self.__grid_size):
            for c in range(self.__grid_size):
                color = COLORS[self.grid[r][c]]
                pygame.draw.rect(screen, color, (c * self.__tile_size, r * self.__tile_size, self.__tile_size,
                                                 self.__tile_size))

                if self.grid[r][c] == 'P':
                    screen.blit(self.__pig_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'G':
                    screen.blit(self.__egg_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'Q':
                    screen.blit(self.__queen_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'R':
                    screen.blit(self.__rock_with_background, (c * self.__tile_size, r * self.__tile_size))

        for r in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (0, r * self.__tile_size), (self.__grid_size * self.__tile_size,
                                                                            r * self.__tile_size), 2)
        for c in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (c * self.__tile_size, 0), (c * self.__tile_size,
                                                                            self.__grid_size * self.__tile_size), 2)

        agent_row, agent_col = self.__agent_pos
        screen.blit(self.__agent_image, (agent_col * self.__tile_size, agent_row * self.__tile_size))

    def euclidean_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def k_means(self, points, k, max_iter=100):
        centroids = points[:k]
        clusters = [[] for _ in range(k)]

        for _ in range(max_iter):
            # Assign points to nearest centroid
            for point in points:
                distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(point)

            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if len(cluster) == 0:
                    new_centroids.append(random.choice(points))
                else:
                    new_centroid = tuple([sum(coord) / len(coord) for coord in zip(*cluster)])
                    new_centroids.append(new_centroid)

            if new_centroids == centroids:
                break
            centroids = new_centroids
            clusters = [[] for _ in range(k)]

        return clusters, centroids

    def best_clustering(self, points):
        best_score = -1  # To maximize silhouette score
        best_clusters = None
        best_centroids = None
        best_k = 0

        # بررسی برای k = 2، 3، و 4
        for k in [2, 3, 4]:
            clusters, centroids = self.k_means(points, k)
            flat_points = [point for cluster in clusters for point in cluster]
            cluster_labels = [i for i, cluster in enumerate(clusters) for _ in cluster]

            score = silhouette_score(flat_points, cluster_labels)  # محاسبه ضریب سیلیک
            if score > best_score:
                best_score = score
                best_clusters = clusters
                best_centroids = centroids
                best_k = k

        centroid_to_cluster = {}
        # نمایش خوشه‌ها و مراکز بهترین تقسیم‌بندی
        print(f"Best clustering with {best_k} clusters:")
        for i, cluster in enumerate(best_clusters):
            print(f"Cluster {i + 1}: {cluster}, Centroid: {best_centroids[i]}")
            centroid_to_cluster[best_centroids[i]] = cluster
        return best_centroids, best_clusters, centroid_to_cluster

    def order_of_v_tables(self, current_state):

        list_of_clusters = self.cluster_points
        goal_point = (7, 7)
        perms = itertools.permutations(list_of_clusters)
        best_permutate = ()
        ans1 = 10000
        for perm in perms:
            sum1 = 0
            for i in range(len(perm) - 1):
                sum1 += abs(perm[i + 1][0] - perm[i][0]) + abs(perm[i + 1][1] - perm[i][1])
            sum1 += abs(perm[len(perm) - 1][0] - goal_point[0]) + abs(perm[len(perm) - 1][1] - goal_point[1])
            sum1 += abs(perm[0][0] - current_state[0]) + abs(perm[0][1] - current_state[1])
            if sum1 < ans1:
                best_permutate = perm
                ans1 = sum1
        best_permutate += (goal_point,)
        return best_permutate

    def get_reward_table_for(self, center):
        reward_map = [[-1 for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]
        if center == (7, 7):
            reward_map[7][7] = 2000
            for row in range(self.__grid_size):
                for col in range(self.__grid_size):
                    tile = self.grid[row][col]
                    if tile == 'R':
                        reward_map[row][col] = -100
            return reward_map
        for row in range(self.__grid_size):
            for col in range(self.__grid_size):
                tile = self.grid[row][col]
                base_reward = -1  # Default reward for empty tiles
                if (row, col) in self.centroid_to_cluster_dict[center]:
                    base_reward = 200
                # if tile == 'P':
                #     base_reward = 200
                elif tile == 'Q':
                    base_reward = -275
                # elif tile == 'G':
                #     base_reward = 1500
                # elif tile == 'R':
                #     base_reward = -50

                # Apply potential-based shaping
                # current_potential = self.potential_function((row, col))
                reward_map[row][col] = base_reward  # + current_potential
        return reward_map

    def get_v_tables(self, best_permutate):
        list_of_action_models = []
        for center in best_permutate:
            print(center)
            reward_table = self.get_reward_table_for(center)
            action_model = self.actions_model(reward_table)
            list_of_action_models.append(action_model)
        return list_of_action_models

    def actions_model(self, reward_map):
        values = np.zeros((self.__grid_size, self.__grid_size))
        gamma = 0.84  # Discount factor
        epsilon = 0.01  # Convergence threshold
        max_iterations = 2500
        iteration = 0
        actions_taken = np.full((self.__grid_size, self.__grid_size), -1)
        last = False
        if reward_map[7][7] == 2000:
            last = True

        while True:
            delta = 0
            new_values = np.copy(values)
            for x in range(self.__grid_size):
                for y in range(self.__grid_size):
                    if self.grid[x][y] == 'R':
                        continue  # Skip rocks
                    state = (x, y)
                    v = values[x, y]
                    max_value = float('-inf')
                    best_action = None
                    for action in [0, 1, 2, 3]:
                        value = 0
                        for prob, next_state, reward in self.transition_table[state][action]:
                            nx, ny = next_state
                            value += prob * (reward_map[state[0]][state[1]] + gamma * values[nx, ny])
                        if value > max_value:
                            max_value = value
                            best_action = action
                    new_values[x, y] = max_value
                    delta = max(delta, abs(v - new_values[x, y]))
                    actions_taken[x, y] = best_action
            values = new_values
            iteration += 1
            if delta < epsilon or iteration >= max_iterations:
                break

        return actions_taken

    def potential_function(self, state):
        # Define a potential function φ(s)
        # For example, use the negative Euclidean distance to the nearest cluster center
        max_potential = 50
        x, y = state
        distances = [self.euclidean_distance((x, y), center) for center in self.cluster_points]
        return max_potential * math.exp(-min(distances))

    def reward_function(self):
        # implement this function
        """it returns a 8x8 matrix
        [[-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -400, -1, -1, -1, 180, -1, -1],
        ...'"""
        reward_map = [[-1 for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]
        for row in range(self.__grid_size):
            for col in range(self.__grid_size):
                tile = self.grid[row][col]
                base_reward = -1  # Default reward for empty tiles
                if tile == 'P':
                    base_reward = 200
                elif tile == 'Q':
                    base_reward = -275
                elif tile == 'G':
                    base_reward = 1500
                elif tile == 'R':
                    base_reward = -50

                # Apply potential-based shaping
                current_potential = self.potential_function((row, col))
                reward_map[row][col] = base_reward + current_potential
        return reward_map

    @classmethod
    def __calculate_transition_model(cls, grid_size, actions_prob, reward_map):
        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1)  # Right
        }

        neighbors = {
            0: [2, 3],  # Up -> Left and Right
            1: [2, 3],  # Down -> Left and Right
            2: [0, 1],  # Left -> Up and Down
            3: [0, 1]  # Right -> Up and Down
        }

        transition_table = {}

        for row in range(grid_size):
            for col in range(grid_size):
                state = (row, col)
                transition_table[state] = {}

                for action in range(4):
                    transition_table[state][action] = []

                    intended_move = actions[action]
                    next_state = (row + intended_move[0], col + intended_move[1])

                    if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
                        reward = reward_map[next_state[0]][next_state[1]]
                        intended_probability = actions_prob[(next_state[0], next_state[1])][action]['intended']
                        transition_table[state][action].append((intended_probability, next_state, reward))
                    else:
                        intended_probability = actions_prob[state][action]['intended']
                        transition_table[state][action].append(
                            (intended_probability, state, reward_map[row][col]))

                    for neighbor_action in neighbors[action]:
                        neighbor_move = actions[neighbor_action]
                        next_state = (row + neighbor_move[0], col + neighbor_move[1])

                        if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
                            reward = reward_map[next_state[0]][next_state[1]]
                            neighbor_probability = actions_prob[(next_state[0], next_state[1])][action]['neighbor']
                            transition_table[state][action].append((neighbor_probability, next_state, reward))
                        else:
                            neighbor_probability = actions_prob[state][action]['neighbor']
                            transition_table[state][action].append(
                                (neighbor_probability, state, reward_map[row][col]))

        return transition_table

    @classmethod
    def __is_path_exists(cls, grid, start, goal):
        grid_size = len(grid)
        visited = set()

        def dfs(x, y):
            if (x, y) == goal:
                return True
            visited.add((x, y))

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid_size and 0 <= ny < grid_size and
                        (nx, ny) not in visited and grid[nx][ny] != 'R'):
                    if dfs(nx, ny):
                        return True
            return False

        return dfs(start[0], start[1])

    def __generate_probability_dict(self):
        probability_dict = {}

        for row in range(self.__grid_size):
            for col in range(self.__grid_size):
                state = (row, col)
                probability_dict[state] = {}

                for action in range(4):
                    intended_prob = random.uniform(0.60, 0.80)
                    remaining_prob = 1 - intended_prob
                    neighbor_prob = remaining_prob / 2

                    probability_dict[state][action] = {
                        'intended': intended_prob,
                        'neighbor': neighbor_prob}
        return probability_dict


if __name__ == "__main__":

    FPS = 100000
    env = AngryBirds()
    screen, clock = PygameInit.initialization()
    state = env.reset()
    env.best_clustering(env.pig_points_for_cluster)
    sum_of_all = 0
    for _ in range(5):
        running = True
        current = 0
        count = 0
        moves = 0
        v_tables = env.get_v_tables(env.order_of_v_tables(state))
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.render(screen)
            s = int(moves / 50)
            if s > len(v_tables) - 1:
                s = len(v_tables) - 1
            action_table = v_tables[s]
            moves += 1
            action = action_table[state]
            next_state, probability, reward_episode, done = env.step(action)
            state = next_state
            current += reward_episode
            if done:
                moves = 0
                env.best_clustering(env.pig_points_for_cluster)
                print(f"current_score: {current}")
                sum_of_all += current
                state = env.reset()
                # running = False
                break

            pygame.display.flip()
            clock.tick(FPS)
    print(sum_of_all / 5)
    pygame.quit()
