import numpy as np
import pygame
from game import AngryGame, PygameInit


class MiniMax:
    _actions_limit = 200

    def __init__(self, environment):
        self.env = environment
        self._goal = 200 * len(environment.get_egg_coordinate(environment.grid))
        self.endpoint = environment.get_slingshot_position(environment.grid)
        self._is_win = environment.is_win
        self._is_lose = environment.is_lose
        self._calculate_score = environment.calculate_score

    @staticmethod
    def distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _heuristic_function(self, grid, num_actions):
        hen_position = self.env.get_hen_position(grid)  # the escape agent
        bird_position = self.env.get_bird_position(grid)  # the chase agent
        queen_position = self.env.get_queen_position(grid)  # the hostile agent
        egg_positions = self.env.get_egg_coordinate(grid)  # list of (x, y) positions
        pig_positions = self.env.get_pig_coordinate(grid)  # list of (x, y) positions
        endpoint = self.endpoint

        hen_score = self._goal - 200 * len(self.env.get_egg_coordinate(self.env.grid))

        finishing_factor = 0 if (hen_score - self._goal) / 1000 < 0 else (hen_score - self._goal) / 1000

        # the bird score if reached appropriate total score
        hen_score += 250 * finishing_factor if self.env.get_slingshot_position(grid) is None else 0

        # the bird score if reached appropriate total score
        bird_score = 600 * finishing_factor if self.env.is_queen_exists(grid) else 0

        # the bird (chase agent) just finish the game, no scores
        total_score = self.env.calculate_score(grid, num_actions)

        # calculate the hen's safety (distance from hostile agent)
        hen_safety = self.distance(hen_position, queen_position)

        # calculate the bird's chase value (inverse distance to hostile agent)
        bird_chase = 10 - self.distance(bird_position, queen_position)

        # calculate proximity to endpoint for the hen
        hen_to_endpoint = finishing_factor * (10 - self.distance(hen_position, endpoint))

        # calculate proximity to eggs for the eggs
        hen_to_eggs = sum(10 - self.distance(hen_position, egg_position) for egg_position in egg_positions)
        # calculate proximity to pigs for the eggs
        hen_to_pigs = sum(10 - self.distance(hen_position, pig_position) for pig_position in pig_positions)

        # early finishing penalty
        early_finish = self._goal - hen_score if self._is_win(grid) else 0

        # losing penalty
        lose_penalty = 1000 if not self.env.is_hen_exists(grid) or num_actions == self._actions_limit else 0

        # calculate final utility
        utility = (
                1 * hen_safety +  # safety distance for the hen
                2 * bird_chase +  # chase efficiency for the bird
                1 * hen_to_endpoint +  # proximity to endpoint
                2 * hen_to_eggs -  # proximity to eggs
                0.1 * hen_to_pigs -  # proximity to eggs
                1 +  # price of action
                1 * early_finish -  # early finish penalty
                1 * lose_penalty  # lose penalty
        )

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
        if self._is_lose(grid, num_actions) or self._is_win(grid):
            return self._calculate_score(grid, num_actions)  # actual game score

        # depth limit reached: combine actual score with heuristic
        if depth == 0:
            return self._calculate_score(grid, num_actions) + self._heuristic_function(grid, num_actions)

        if agent == 'hen':  # hen maximizing agent
            max_score = float('-inf')
            for successor, _ in self.env.generate_hen_successors(grid):
                score = self._alpha_beta_search(successor, num_actions + 1, depth - 1, alpha, beta, 'bird')
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # beta cutoff
            return max_score

        elif agent == 'bird':  # bird maximizing agent
            max_score = float('-inf')
            for successor, _ in self.env.generate_bird_successors(grid):
                score = self._alpha_beta_search(successor, num_actions + 1, depth - 1, alpha, beta, 'queen')
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # beta cutoff
            return max_score

        elif agent == 'queen':  # queen minimizing
            min_score = float('inf')
            for successor, _ in self.env.generate_queen_successors(grid):
                score = self._alpha_beta_search(successor, num_actions, depth - 1, alpha, beta, 'hen')
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break  # alpha cutoff
            return min_score

    def get_best_actions(self, grid, num_actions, turn, depth=5):
        depth *= 3
        _best_action = None
        max_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        get_successors = self.env.generate_bird_successors
        if turn == 0:
            get_successors = self.env.generate_hen_successors

        for successor, _action in get_successors(grid):
            score = self._alpha_beta_search(successor, num_actions + 1, depth, alpha, beta, 'bird')
            if score > max_score:
                max_score = score
                _best_action = _action
            alpha = max(alpha, score)

        return _best_action


    def update_environment(self, environment):
        self.env = environment

if __name__ == "__main__":

    env = AngryGame(template='hard')

    screen, clock = PygameInit.initialization()
    FPS = 6

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
        best_action = minimax.get_best_actions(env.grid, env.num_actions, counter % 3, depth=3)

        if counter % 3 == 0:
            action = best_action
            env.hen_step(action)
            env.render(screen)
            if AngryGame.is_win(env.grid):
                running = False

        if counter % 3 == 1:
            action = best_action
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
