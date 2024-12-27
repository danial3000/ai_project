import pygame
from heapq import heappush, heappop
import copy
import numpy as np
import os
from collections import deque
from queue import PriorityQueue
import csv

#######################################################
#                DONT CHANGE THIS PART                #
#######################################################
COLORS = {
    'T': (143, 188, 143),  # Tile ground
    'Q': (143, 188, 143),  # Queen
    'R': (143, 188, 143),  # Rock
    'H': (143, 188, 143),  # Hen
    'E': (143, 188, 143),  # Egg
    'S': (143, 188, 143),  # Slingshot
    'P': (143, 188, 143)  # Pigs
}

EGG_REWARD = 200
PIG_REWARD = -200
DEFAULT_REWARD = (-1)
LOSE_REWARD = -1000
SLING_REWARD = 400

EGGS = 8
PIGS = 8
QUEEN = 1

MAX_ACTIONS = 150


#######################################################
#                DONT CHANGE THIS PART                #
#######################################################


class PygameInit:

    @classmethod
    def initialization(cls):
        grid_size_x = 10
        grid_size_y = 10
        tile_size = 80

        pygame.init()
        screen = pygame.display.set_mode((grid_size_x * tile_size, grid_size_y * tile_size))
        pygame.display.set_caption("GAME")
        clock = pygame.time.Clock()

        return screen, clock


#######################################################
#                DONT CHANGE THIS PART                #
#######################################################
class AngryGame:
    def __init__(self, template: str):
        self.__grid_size = 10
        self.__tile_size = 80
        self.__template_type = template

        self.__base_grid = self.__generate_grid()
        self.grid = copy.deepcopy(self.__base_grid)
        self.__base_grid = copy.deepcopy(self.grid)

        self.num_actions = 0

        self.__hen_image = pygame.image.load("Env/icons/white bird.png")
        self.__hen_image = pygame.transform.scale(self.__hen_image, (self.__tile_size, self.__tile_size))

        self.__queen_image = pygame.image.load('Env/icons/queen.png')
        self.__queen_image = pygame.transform.scale(self.__queen_image, (self.__tile_size, self.__tile_size))
        self.__queen_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__queen_with_background.fill((143, 188, 143))
        self.__queen_with_background.blit(self.__queen_image, (0, 0))

        self.__pig_image = pygame.image.load('Env/icons/pig.png')
        self.__pig_image = pygame.transform.scale(self.__pig_image, (self.__tile_size, self.__tile_size))
        self.__pig_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__pig_with_background.fill((143, 188, 143))
        self.__pig_with_background.blit(self.__pig_image, (0, 0))

        self.__egg = pygame.image.load('Env/icons/egg.png')
        self.__egg = pygame.transform.scale(self.__egg, (self.__tile_size, self.__tile_size))
        self.__egg_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__egg_with_background.fill((143, 188, 143))
        self.__egg_with_background.blit(self.__egg, (0, 0))

        self.__rock_image = pygame.image.load('Env/icons/rocks.png')
        self.__rock_image = pygame.transform.scale(self.__rock_image, (self.__tile_size, self.__tile_size))
        self.__rock_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__rock_with_background.fill((143, 188, 143))
        self.__rock_with_background.blit(self.__rock_image, (0, 0))

        self.__slingshot_image = pygame.image.load('Env/icons/slingshot.png')
        self.__slingshot_image = pygame.transform.scale(self.__slingshot_image, (self.__tile_size, self.__tile_size))
        self.__slingshot_image_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__slingshot_image_background.fill((143, 188, 143))
        self.__slingshot_image_background.blit(self.__slingshot_image, (0, 0))

    def __generate_grid(self):

        grid = [['T' for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]

        with open(f'Env/templates/{self.__template_type}.txt') as file:
            template_str = file.readlines()

        for i in range(self.__grid_size):
            for j in range(self.__grid_size):
                grid[i][j] = template_str[i][j]

        return grid

    def reset(self):
        self.grid = copy.deepcopy(self.__base_grid)
        self.num_actions = 0

    def hen_step(self, agent_action):
        hen_pos = self.get_hen_position(self.grid)

        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        dx, dy = actions[agent_action]
        new_row = hen_pos[0] + dx
        new_col = hen_pos[1] + dy

        if self.__is_valid_for_hen_position(self.grid, new_row, new_col):
            self.grid[hen_pos[0]][hen_pos[1]] = 'T'
            hen_pos = (new_row, new_col)
            self.grid[hen_pos[0]][hen_pos[1]] = 'H'

            self.num_actions += 1

    def queen_step(self):
        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        if self.is_queen_exists(self.grid):
            queen_pos = self.get_queen_position(self.grid)
            hen_pos = self.get_hen_position(self.grid)

            best_action = None
            min_cost = float('inf')

            for action, (dx, dy) in actions.items():
                new_row, new_col = queen_pos[0] + dx, queen_pos[1] + dy

                if self.__is_valid_for_queen_position(self.grid, new_row, new_col) and \
                        self.grid[new_row][new_col] != 'E':
                    cost = self.__a_star_cost((new_row, new_col), hen_pos)

                    if cost < min_cost:
                        min_cost = cost
                        best_action = action

            if best_action is not None:
                dx, dy = actions[best_action]
                new_row, new_col = queen_pos[0] + dx, queen_pos[1] + dy

                self.grid[queen_pos[0]][queen_pos[1]] = 'T'
                queen_pos = (new_row, new_col)
                self.grid[queen_pos[0]][queen_pos[1]] = 'Q'

    @classmethod
    def generate_hen_successors(cls, grid):
        hen_pos = cls.get_hen_position(grid)
        if not hen_pos:
            return []

        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        successors = []
        for action in actions:
            dx, dy = actions[action]
            new_row, new_col = hen_pos[0] + dx, hen_pos[1] + dy
            if cls.__is_valid_for_hen_position(grid, new_row, new_col):
                successor_grid = copy.deepcopy(grid)

                successor_grid[new_row][new_col] = 'H'

                successor_grid[hen_pos[0]][hen_pos[1]] = 'T'
                successors.append((successor_grid, action))

        return successors

    @classmethod
    def generate_queen_successors(cls, grid):
        queen_pos = cls.get_queen_position(grid)
        if not queen_pos:
            return []

        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        successors = []
        for action in actions:
            dx, dy = actions[action]
            new_row, new_col = queen_pos[0] + dx, queen_pos[1] + dy
            if cls.__is_valid_for_queen_position(grid, new_row, new_col):
                successor_grid = copy.deepcopy(grid)

                successor_grid[new_row][new_col] = 'Q'
                successor_grid[queen_pos[0]][queen_pos[1]] = 'T'
                successors.append((successor_grid, action))

        return successors

    @classmethod
    def get_egg_coordinate(cls, grid):
        food_coordinates = []
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 'E':
                    food_coordinates.append((r, c))
        return food_coordinates

    @classmethod
    def get_pig_coordinate(cls, grid):
        pig_coordinates = []
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 'P':
                    pig_coordinates.append((r, c))
        return pig_coordinates

    @classmethod
    def get_hen_position(cls, grid):
        for r in range(len(grid)):
            for c in range(len(grid)):
                if grid[r][c] == 'H':
                    return tuple([r, c])

    @classmethod
    def get_queen_position(cls, grid):
        for r in range(len(grid)):
            for c in range(len(grid)):
                if grid[r][c] == 'Q':
                    return tuple([r, c])

    @classmethod
    def get_slingshot_position(cls, grid):
        for r in range(len(grid)):
            for c in range(len(grid)):
                if grid[r][c] == 'S':
                    return tuple([r, c])

    @classmethod
    def __check_lose(cls, grid):
        for r in range(len(grid)):
            for c in range(len(grid)):
                if grid[r][c] == 'H':
                    return False
        return True

    @classmethod
    def is_win(cls, grid):
        hen_pos = cls.get_hen_position(grid)
        queen_pos = cls.get_queen_position(grid)
        sling_pos = cls.get_slingshot_position(grid)

        if not sling_pos:
            return True

        if not queen_pos:
            return True

        if hen_pos and sling_pos and hen_pos == sling_pos:
            return True

        return False

    @classmethod
    def is_lose(cls, grid, num_actions):
        return cls.__check_lose(grid) or num_actions >= MAX_ACTIONS

    @classmethod
    def calculate_score(cls, grid, num_actions):

        egg_score = (EGGS - len(cls.get_egg_coordinate(grid))) * EGG_REWARD

        pig_score = (PIGS - len(cls.get_pig_coordinate(grid))) * PIG_REWARD

        actions_score = DEFAULT_REWARD * num_actions

        sling_score = SLING_REWARD if cls.is_win(grid) else 0
        lose_score = LOSE_REWARD if cls.is_lose(grid, num_actions) else 0

        total = egg_score + sling_score + actions_score + pig_score + lose_score

        return total

    @classmethod
    def generate_pos_successors(cls, grid, pos, type):
        hen_pos = pos
        if not hen_pos:
            return []

        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        successors = []
        if type == 'queen-bfs':
            for action in actions:
                dx, dy = actions[action]
                new_row, new_col = hen_pos[0] + dx, hen_pos[1] + dy
                if cls.__is_valid_for_hen_position_in_bfs(grid, new_row, new_col):
                    successors.append((new_row, new_col))

        if type == 'sling-bfs':
            for action in actions:
                dx, dy = actions[action]
                new_row, new_col = hen_pos[0] + dx, hen_pos[1] + dy
                if cls.__is_valid_for_sling_position_in_bfs(grid, new_row, new_col):
                    successors.append((new_row, new_col))

        return successors

    @classmethod
    def get_egg_distance_from_bfs(cls, grid, egg_loc, current_loc):
        queue = deque()
        visited = set()
        hen_pos = current_loc
        queue.append((hen_pos[0], hen_pos[1]))
        depth = 0
        while queue:
            for i in range(len(queue)):
                q = queue.popleft()
                visited.add(q)
                if (q[0], q[1]) == egg_loc:
                    return depth
                g = cls.generate_pos_successors(grid, q, 'queen-bfs')
                for q_successor in g:
                    if q_successor not in visited:
                        queue.append((q_successor[0], q_successor[1]))
                        visited.add(q_successor)
            depth += 1

    @classmethod
    def get_egg_position_ucs(cls, grid, loc):
        priority_queue = []
        visited = set()
        hen_pos = cls.get_hen_position(grid)

        heappush(priority_queue, (0, (hen_pos[0], hen_pos[1])))

        while priority_queue:
            current_cost, current_node = heappop(priority_queue)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == loc:
                return current_cost

            successors = cls.generate_pos_successors(grid, current_node, 'queen-bfs')
            for successor in successors:
                if successor not in visited:
                    new_cost = current_cost + 1
                    if grid[successor[0]][successor[1]] == 'P':
                        new_cost += 200

                    heappush(priority_queue, (new_cost, successor))

        return 1000

    @classmethod
    def get_sling_position_bfs(cls, grid):
        print(cls.print_grid(grid))
        priority_queue = []
        visited = set()
        hen_pos = cls.get_hen_position(grid)

        heappush(priority_queue, (0, (hen_pos[0], hen_pos[1])))  # (هزینه, موقعیت)

        while priority_queue:
            # کم‌هزینه‌ترین گره را از صف اولویت‌دار خارج می‌کنیم
            current_cost, current_node = heappop(priority_queue)

            if current_node in visited:
                continue

            visited.add(current_node)

            if grid[current_node[0]][current_node[1]] == 'S':
                return current_cost

            successors = cls.generate_pos_successors(grid, current_node, 'sling-bfs')
            for successor in successors:
                if successor not in visited:
                    new_cost = current_cost + 1
                    if grid[successor[0]][successor[1]] == 'P':
                        new_cost += 200

                    heappush(priority_queue, (new_cost, successor))

        return 1000

    @classmethod
    def correct_grid(cls, temp_grid, hen_pos, queen_pos):
        if hen_pos[1] == queen_pos[1]:
            up = queen_pos[1]
            down = queen_pos[1]
            for i in range(10):
                up += 1
                down -= 1
                if up > 9 or temp_grid[queen_pos[0]][up] == 'R':
                    up -= 1
                if down < 0 or temp_grid[queen_pos[0]][down] == 'R':
                    down += 1
                if temp_grid[queen_pos[0]][up] != 'S' and temp_grid[queen_pos[0]][up] != 'P' and \
                        temp_grid[queen_pos[0]][up] != 'Q':
                    temp_grid[queen_pos[0]][up] = 'R'
                if temp_grid[queen_pos[0]][down] != 'S' and temp_grid[queen_pos[0]][down] != 'P' and \
                        temp_grid[queen_pos[0]][down] != 'Q':
                    temp_grid[queen_pos[0]][down] = 'R'
        else:
            up = queen_pos[0]
            down = queen_pos[0]
            for i in range(10):
                up += 1
                down -= 1
                if up > 9 or temp_grid[up][queen_pos[1]] == 'R':
                    up -= 1
                if down < 0 or temp_grid[down][queen_pos[1]] == 'R':
                    down += 1
                if temp_grid[up][queen_pos[1]] != 'S' and temp_grid[up][queen_pos[1]] != 'P' and \
                        temp_grid[up][queen_pos[1]] != 'Q':
                    temp_grid[up][queen_pos[1]] = 'R'
                if temp_grid[down][queen_pos[1]] != 'S' and temp_grid[down][queen_pos[1]] != 'P' and \
                        temp_grid[down][queen_pos[1]] != 'Q':
                    temp_grid[down][queen_pos[1]] = 'R'
        return temp_grid

    # Added: Heuristic evaluation for intermediate states
    @classmethod
    def heuristic_evaluation(cls, grid, num_actions):

        base_score = cls.calculate_score(grid, num_actions)

        # hen_successors = cls.generate_hen_successors(grid)
        #
        # queen_successors = cls.generate_queen_successors(grid)

        hen_pos = cls.get_hen_position(grid)

        if hen_pos is None:
            print(hen_pos)

        eggs = cls.get_egg_coordinate(grid)

        queen_pos = cls.get_queen_position(grid)

        sum_dists = 0
        max_dist = -1
        min_dist = 1000000
        for egg in eggs:
            dist = cls.get_egg_position_ucs(grid, (egg[0], egg[1]))
            sum_dists += dist
            min_dist = min(min_dist, dist)
            max_dist = max(max_dist, dist)

        queen_dis = abs(queen_pos[0] - hen_pos[0]) + abs(queen_pos[1] - hen_pos[1])

        sling_position = cls.get_slingshot_position(grid)
        sling_eated = False
        if sling_position is not None:
            sling_eated = False
        else:
            sling_eated = True
        sling_dist = cls.get_sling_position_bfs(grid)
        if base_score > 1200 or (sling_dist != 1000 and num_actions + sling_dist >= 140):
            if not sling_eated:
                if abs(hen_pos[0] - queen_pos[0]) + abs(hen_pos[1] - queen_pos[1]) == 1:

                    temp_grid = cls.correct_grid(grid, hen_pos, queen_pos)

                    sling_dist = cls.get_sling_position_bfs(temp_grid)
                heuristic_score = base_score - sling_dist  # + (3 * pow(queen_dis, 2))
                return heuristic_score
            else:
                if base_score >= 1500 or (sling_dist != 1000 and num_actions + sling_dist >= 140):
                    heuristic_score = base_score - (len(eggs) * sum_dists) + (3 * pow(queen_dis, 2))
                else:
                    heuristic_score = base_score - (len(eggs) * sum_dists) + (3 * pow(queen_dis, 2)) - 400
                return heuristic_score
        else:
            if not sling_eated:
                heuristic_score = base_score - (len(eggs) * sum_dists) + (3 * pow(queen_dis, 2))
                return heuristic_score
            else:
                heuristic_score = base_score - (len(eggs) * sum_dists) + (3 * pow(queen_dis, 2)) - 600
                return heuristic_score

    def render(self, screen):
        for r in range(self.__grid_size):
            for c in range(self.__grid_size):
                color = COLORS[self.grid[r][c]]
                pygame.draw.rect(screen, color, (c * self.__tile_size, r * self.__tile_size, self.__tile_size,
                                                 self.__tile_size))

                if self.grid[r][c] == 'H':
                    screen.blit(self.__hen_image, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'Q':
                    screen.blit(self.__queen_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'R':
                    screen.blit(self.__rock_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'E':
                    screen.blit(self.__egg_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'S':
                    screen.blit(self.__slingshot_image_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'P':
                    screen.blit(self.__pig_with_background, (c * self.__tile_size, r * self.__tile_size))

        for r in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (0, r * self.__tile_size), (self.__grid_size * self.__tile_size,
                                                                            r * self.__tile_size), 2)
        for c in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (c * self.__tile_size, 0), (c * self.__tile_size,
                                                                            self.__grid_size * self.__tile_size), 2)

    @classmethod
    def __is_valid_for_queen_position(cls, grid, new_row, new_col):
        return (
                0 <= new_row < len(grid)
                and 0 <= new_col < len(grid[0])
                and grid[new_row][new_col] != 'R'
                and grid[new_row][new_col] != 'S'
                and grid[new_row][new_col] != 'P')

    @classmethod
    def __is_valid_for_hen_position_in_bfs(cls, grid, new_row, new_col):

        return (
                0 <= new_row < len(grid)
                and 0 <= new_col < len(grid)
                and grid[new_row][new_col] != 'R'
                and grid[new_row][new_col] != 'P'
                and grid[new_row][new_col] != 'S'
        )

    @classmethod
    def __is_valid_for_sling_position_in_bfs(cls, grid, new_row, new_col):

        return (
                0 <= new_row < len(grid)
                and 0 <= new_col < len(grid)
                and grid[new_row][new_col] != 'Q'
                and grid[new_row][new_col] != 'R'
        )

    @classmethod
    def __is_valid_for_hen_position(cls, grid, new_row, new_col):

        return (
                0 <= new_row < len(grid)
                and 0 <= new_col < len(grid)
                and grid[new_row][new_col] != 'Q'
                and grid[new_row][new_col] != 'R'
        )

    @classmethod
    def is_queen_exists(cls, grid):
        for r in range(len(grid)):
            for c in range(len(grid)):
                if grid[r][c] == 'Q':
                    return True
        return False

    @classmethod
    def is_hen_exists(cls, grid):
        for r in range(len(grid)):
            for c in range(len(grid)):
                if grid[r][c] == 'H':
                    return True
        return False

    @classmethod
    def print_grid(cls, grid):
        printed_grid = ''

        for r in range(len(grid)):
            printed_grid += '\n'
            for c in range(len(grid)):
                printed_grid += grid[r][c]

        return printed_grid + '\n'

    def __a_star_cost(self, start, goal):

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heappush(open_set, (0, start))
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                return g_score[current]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.__is_valid_for_queen_position(self.grid, neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return float('inf')

    @staticmethod
    def state_to_tuple(grid):
        return tuple(tuple(row) for row in grid)

    def minimax(self, grid, num_actions, depth, maximizing_player, alpha=float('-inf'), beta=float('inf'),
                visited=None):

        if visited is None:
            visited = set()

        current_state = self.state_to_tuple(grid)
        if current_state in visited:
            score = self.calculate_score(grid, num_actions)
            return score, None
        visited.add(current_state)

        if depth == 0 or self.is_win(grid) or self.is_lose(grid, num_actions):
            if self.is_lose(grid, num_actions):
                base_score = self.calculate_score(grid, num_actions) - 10000
                return base_score, None
            else:
                score = self.heuristic_evaluation(grid, num_actions)  # CHANGED to heuristic evaluation
                return score, None

        if maximizing_player:
            max_eval = float('-inf')
            best_action = None
            hen_successors = self.generate_hen_successors(grid)
            if not hen_successors:
                score = self.heuristic_evaluation(grid, num_actions)
                return score, None

            def min_distance_to_egg(g):
                hen_p = self.get_hen_position(g)
                eggs = self.get_egg_coordinate(g)
                if not eggs or not hen_p:
                    return float('inf')
                return min(abs(hen_p[0] - e[0]) + abs(hen_p[1] - e[1]) for e in eggs)

            for successor, action in hen_successors:
                eval, _ = self.minimax(successor, num_actions + 1, depth - 1, False, alpha, beta,
                                       visited=visited.copy())
                action_label = self.action_to_label(action)

                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                    best_dist = min_distance_to_egg(successor)
                elif eval == max_eval:
                    # Tie-Breaker:
                    current_dist = min_distance_to_egg(successor)
                    if current_dist < best_dist:
                        best_dist = current_dist
                        best_action = action

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cut-off

            return max_eval, best_action
        else:
            min_eval = float('inf')
            best_action = None
            queen_successors = self.generate_queen_successors(grid)

            if not queen_successors:
                score = self.heuristic_evaluation(grid, num_actions)
                return score, None

            for successor, action in queen_successors:
                eval, _ = self.minimax(successor, num_actions, depth - 1, True, alpha, beta,
                                       visited=visited.copy())
                action_label = self.action_to_label(action)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval, best_action

    def action_to_label(self, action):
        actions = {
            0: "Up",
            1: "Down",
            2: "Left",
            3: "Right"
        }
        return actions.get(action, "Unknown")

    def choose_best_hen_action_with_tree(self, depth=3, move_number=0):
        score, action = self.minimax(self.grid, self.num_actions, depth, True)
        filename = f'tree{move_number}.jpg'
        return action

    def choose_best_hen_action(self, depth=3):
        score, action = self.minimax(self.grid, self.num_actions, depth, True)
        return action


if __name__ == "__main__":

    env = AngryGame(template='test9')

    screen, clock = PygameInit.initialization()
    FPS = 5000

    env.reset()
    counter = 0

    running = True
    while running:
        if AngryGame.is_win(env.grid) or AngryGame.is_lose(env.grid, env.num_actions):
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if counter % 2 == 0:
            move_number = counter // 2
            action = env.choose_best_hen_action_with_tree(depth=7, move_number=move_number)
            if action is not None:
                env.hen_step(action)
            else:
                action = np.random.choice([0, 1, 2, 3])
                env.hen_step(action)
            env.render(screen)
            if AngryGame.is_win(env.grid):
                running = False

        if counter % 2 == 1:
            env.queen_step()
            env.render(screen)
            if AngryGame.is_lose(env.grid, env.num_actions):
                running = False

        counter += 1
        pygame.display.flip()
        clock.tick(FPS)
        print(f'Current Score == {AngryGame.calculate_score(env.grid, env.num_actions)}')

    pygame.quit()
