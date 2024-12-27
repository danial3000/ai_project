"""
Project Title: Game

Department:
    Artificial Intelligence Department
    Faculty of Computer Engineering
    University of Isfahan
    November 20, 2024

Supervisor:
    Dr. Hossein Karshenas(h.karshenas@eng.ui.ac.ir) - Professor
    Pouria Sameti(pouria.sameti2002@mehr.ui.ac.ir) - Teaching Assistant

Project Overview:
    This project involves the creation of an interactive game environment designed
    specifically to implement and test game tree algorithms such as Minimax. The
    environment simulates a 10x10 grid-based  game where various entities, including
    a hen, a queen, pigs, eggs, rocks, and a slingshot, interact dynamically.
    These entities introduce complex scenarios involving rewards, penalties, and
    decision-making challenges. The environment is implemented in Python using
    Pygame and features diverse rewards, and win/lose conditions, providing a rich
    testing ground for AI algorithms. By integrating game tree techniques,the project
    aims to analyze optimal decision-making under different game states.

Objectives:
    -Provide a platform for experimenting with game tree enhancements and extensions.
    -Evaluate the effectiveness of the algorithms in decision-making for various entities (e.g., Hen, Queen).
    -Implement game tree algorithms in the environment


Licensing Information:
    -You are free to use or extend these projects for educational purposes.
"""


import pygame
import heapq
import copy

#######################################################
#                DONT CHANGE THIS PART                #
#######################################################
COLORS = {
    'T': (143, 188, 143),    # Tile ground
    'Q': (143, 188, 143),    # Queen
    'R': (143, 188, 143),    # Rock
    'H': (143, 188, 143),    # Hen
    'E': (143, 188, 143),    # Egg
    'S': (143, 188, 143),    # slingshot
    'P': (143, 188, 143)     # Pigs
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
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
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
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
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
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
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
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
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

        return egg_score + sling_score + actions_score + pig_score + lose_score

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
                and grid[new_row][new_col] != 'P'
                and grid[new_row][new_col] != 'E')

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
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

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
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return float('inf')
