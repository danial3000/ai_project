import random

import numpy as np
import pygame

from game import AngryGame, PygameInit


class MiniMax:
    _max_util = 0

    _actions_limit = 200
    _max_distance = 18

    _best_hen_action = 0
    _best_bird_action = 0

    def __init__(self, environment):
        self.env = environment
        self._goal = len(environment.get_egg_coordinate(environment.grid))
        self.endpoint = environment.get_slingshot_position(environment.grid)
        self._is_win = environment.is_win
        self._is_lose = environment.is_lose
        self._calculate_score = environment.calculate_score

        self._get_pig_coordinate = environment.get_pig_coordinate
        self._get_egg_coordinate = environment.get_egg_coordinate

        self._get_hen_position = self.env.get_hen_position
        self._get_bird_position = self.env.get_bird_position
        self._get_queen_position = self.env.get_queen_position


    @staticmethod
    def distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


    def _bird_heuristic_function(self, grid, num_actions):
        hen_position = self._get_hen_position(grid)  # the escape agent
        bird_position = self._get_bird_position(grid)  # the chase agent
        queen_position = self._get_queen_position(grid)  # the hostile agent

        chasing_score = self._max_distance - self.distance(bird_position, queen_position)
        escaping_score = self.distance(hen_position, queen_position)

        utility = chasing_score + escaping_score

        return utility


    def _heuristic_function(self, grid, num_actions):
        egg_positions = list(self._get_egg_coordinate(grid))  # list of (x, y) positions
        # print('finishing factor:', finishing_factor)
        finishing_factor = len(egg_positions) == 0


        pig_positions = self._get_pig_coordinate(grid)  # list of (x, y) positions
        hen_position = self._get_hen_position(grid)  # the escape agent
        bird_position = self._get_bird_position(grid)  # the chase agent
        queen_position = self._get_queen_position(grid)  # the hostile agent
        endpoint = self.endpoint

        egg_score = self._goal - len(self._get_egg_coordinate(self.env.grid))
        # print('egg score:', egg_score)

        # total score
        # total_score = self.env.calculate_score(grid, num_actions)

        # the bird score if reached appropriate total score
        # bird_score = 600 if self.env.is_queen_exists(grid) and total_score > 200 * (self._goal - 2) else 0
        # print('bird score:', bird_score)
        # calculate the hen's safety (distance from hostile agent)
        hen_safety = self.distance(hen_position, queen_position)
        # print('hen safety:', hen_safety)
        # calculate the bird's chase value (inverse distance to hostile agent)
        bird_chase = self._max_distance - self.distance(bird_position, queen_position)
        # print('bird chase:', bird_chase)
        # calculate proximity to endpoint for the hen
        hen_to_endpoint = finishing_factor * (self._max_distance - self.distance(hen_position, endpoint))
        # print('hen_to_endpoint:', hen_to_endpoint)
        # calculate proximity to eggs for the eggs
        hen_to_eggs = sum(self._max_distance - self.distance(hen_position, egg_position) for egg_position in egg_positions)
        # calculate proximity to pigs for the eggs
        hen_to_pigs = min(self._max_distance - self.distance(hen_position, pig_position) for pig_position in pig_positions)
        # print('hen_to_pigs:', hen_to_pigs, 'hen_to_eggs:', hen_to_eggs)
        # early finishing penalty
        # early_finish = self._goal - egg_score if self._is_win(grid) else 0
        # print('early_finish:', early_finish)
        # losing penalty
        # lose_penalty = 1000 if not self.env.is_hen_exists(grid) or num_actions == self._actions_limit else 0
        #print('lose penalty', lose_penalty)
        # calculate final utility
        utility = (
                3 * hen_safety +  # safety distance for the hen
                1 * bird_chase +  # chase efficiency for the bird
                1 * hen_to_endpoint +  # proximity to endpoint
                2 * hen_to_eggs -  # proximity to eggs
                0.1 * hen_to_pigs -  # proximity to eggs
                1  # price of action
        )
        # print('utility:', utility)
        return utility

    def _queen_heuristic(self, grid, queen_position, hen_position, bird_position):
        # distance metrics
        distance_to_hen = self.distance(queen_position, hen_position)
        distance_to_bird = self.distance(queen_position, bird_position)

        # avoid bird, chase hen
        utility = (
                10 * distance_to_hen +  # Move closer to the hen
                -1 * distance_to_bird  # Move away from the bird
        )
        return utility

    def _alpha_beta_search(self, grid, num_actions, depth, alpha=float('-inf'), beta=float('inf'), agent='hen'):

        # terminal states: return actual score
        if self._is_lose(grid, num_actions):
            return self._calculate_score(grid, num_actions), None, None  # actual game score
        elif self._is_win(grid):
            score = self._calculate_score(grid, num_actions)
            to_get_score = 200 * len(self._get_egg_coordinate(grid))
            if to_get_score > 0:
                print()
                return score - to_get_score, None, None
            else:
                return score, None, None  # avoid early ending

        # depth limit reached: combine actual score with heuristic
        if depth == 0:
            return self._calculate_score(grid, num_actions) + self._heuristic_function(grid, num_actions), None, None

        hen_action = []
        bird_action = []

        if agent == 'hen':  # hen maximizing agent
            max_score = float('-inf')
            for successor, ACTION in self.env.generate_hen_successors(grid):
                score, _, bird_action = self._alpha_beta_search(successor, num_actions + 1, depth - 1, alpha, beta, 'bird')
                if score > max_score:
                    max_score = score
                    hen_action = [ACTION]
                if alpha < score:
                    alpha = score
                elif score == max_score:
                    hen_action.append(ACTION)
            return max_score, hen_action, bird_action

        elif agent == 'bird':  # bird maximizing agent
            max_score = float('-inf')
            bird_successors = self.env.generate_bird_successors(grid)
            if self.distance(self._get_bird_position(grid), self._get_queen_position(grid)) % 2 == 0:
                bird_successors = []
                pos = self._get_bird_position(grid)
                if pos[0] == 0:
                    bird_successors.append((grid, 2))
                elif grid[pos[0] - 1][pos[1]] == 'R' or 'P':
                    bird_successors.append((grid, 2))
                if pos[0] == 9:
                    bird_successors.append((grid, 3))
                elif grid[pos[0] + 1][pos[1]] == 'R' or 'P':
                    bird_successors.append((grid, 3))
                if pos[1] == 0:
                    bird_successors.append((grid, 0))
                elif grid[pos[0]][pos[1] - 1] == 'R' or 'P':
                    bird_successors.append((grid, 0))
                if pos[1] == 9:
                    bird_successors.append((grid, 1))
                elif grid[pos[0]][pos[1] + 1] == 'R' or 'P':
                    bird_successors.append((grid, 1))
            for successor, ACTION in bird_successors:
                score, _, _ = self._alpha_beta_search(successor, num_actions + 1, depth - 1, alpha, beta, 'queen')
                if score > max_score:
                    max_score = score
                    bird_action = [ACTION]
                if alpha < score:
                    alpha = score
                elif score == max_score:
                    bird_action.append(ACTION)
            return max_score, [], bird_action

        elif agent == 'queen':  # queen minimizing
            min_score = float('inf')
            for successor, _ in self.env.generate_queen_successors(grid):
                score, _, _ = self._alpha_beta_search(successor, num_actions, depth - 1, alpha, beta, 'hen')
                if score < min_score:
                    min_score = score
                if beta < score:
                    beta = score
            return min_score, [], []

    def get_best_actions(self, grid, num_actions, turn, depth=5):
        if turn == 1:
            return self._best_hen_action, self._best_bird_action
        depth *= 3

        alpha = float('-inf')
        beta = float('inf')

        score, hen_action_list, bird_action_list = self._alpha_beta_search(grid, num_actions, depth, alpha, beta, 'hen')

        self._best_hen_action, self._best_bird_action = random.choice(hen_action_list), random.choice(bird_action_list)

        return self._best_hen_action, self._best_bird_action


    def update_environment(self, environment):
        self.env = environment


if __name__ == "__main__":

    env = AngryGame(template='hard')

    screen, clock = PygameInit.initialization()
    FPS = 8

    DEPTH = 3

    env.reset()
    minimax = MiniMax(environment=env)

    counter = 0
    running = True
    while running:
        if AngryGame.is_win(env.grid) or AngryGame.is_lose(env.grid, env.num_actions):
           running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        minimax.update_environment(env)
        best_hen_action, best_bird_action = minimax.get_best_actions(env.grid, env.num_actions, counter, depth=DEPTH)

        if counter % 3 == 0:
            action = best_hen_action
            env.hen_step(action)
            env.render(screen)
            if AngryGame.is_win(env.grid):
                running = False

        if counter % 3 == 1:
            action = best_bird_action
            env.bird_step(action)
            env.render(screen)
            if AngryGame.is_win(env.grid):
                running = False

        if counter % 3 == 2:
            env.queen_step()
            env.render(screen)
            if AngryGame.is_lose(env.grid, env.num_actions):
                running = False

        counter += 1
        pygame.display.flip()
        clock.tick(FPS)
        print(f'Current Score == {AngryGame.calculate_score(env.grid, env.num_actions)}')

    pygame.quit()
