import numpy as np
import pygame

from game import AngryGame, PygameInit


class MiniMax:
    _actions_limit = 200

    def __init__(self, environment):
        self.env = environment
        self._goal = len(environment.get_egg_coordinate(environment.grid))
        self.endpoint = environment.get_slingshot_position(environment.grid)
        self._is_win = environment.is_win
        self._is_lose = environment.is_lose
        self._calculate_score = environment.calculate_score

    @staticmethod
    def distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])



    def _heuristic_function(self, grid, num_actions):
        max_distance = 18
        egg_positions = self.env.get_egg_coordinate(grid)  # list of (x, y) positions
        #if len(egg_positions) == 0:


        pig_positions = self.env.get_pig_coordinate(grid)  # list of (x, y) positions
        hen_position = self.env.get_hen_position(grid)  # the escape agent
        bird_position = self.env.get_bird_position(grid)  # the chase agent
        queen_position = self.env.get_queen_position(grid)  # the hostile agent
        endpoint = self.endpoint

        egg_score = self._goal - len(self.env.get_egg_coordinate(self.env.grid))
        print('egg score:', egg_score)
        finishing_factor = 1 if len(egg_positions) == 0 else 0
        print('finishing factore:', finishing_factor)
        # the bird score if reached appropriate total score
        egg_score += 250 * finishing_factor if self.env.get_slingshot_position(grid) is None else 0

        # total score
        total_score = self.env.calculate_score(grid, num_actions)

        # the bird score if reached appropriate total score
        bird_score = 600 if self.env.is_queen_exists(grid) and total_score > 200 * (self._goal - 2) else 0
        print('bird score:', bird_score)
        # calculate the hen's safety (distance from hostile agent)
        hen_safety = self.distance(hen_position, queen_position)
        print('hen safety:', hen_safety)
        # calculate the bird's chase value (inverse distance to hostile agent)
        bird_chase = max_distance - self.distance(bird_position, queen_position)
        print('bird chase:', bird_chase)
        # calculate proximity to endpoint for the hen
        hen_to_endpoint = finishing_factor * (max_distance - self.distance(hen_position, endpoint))
        print('hen_to_endpoint:', hen_to_endpoint)
        # calculate proximity to eggs for the eggs
        hen_to_eggs = sum(max_distance - self.distance(hen_position, egg_position) for egg_position in egg_positions)
        # calculate proximity to pigs for the eggs
        hen_to_pigs = min(max_distance - self.distance(hen_position, pig_position) for pig_position in pig_positions)
        print('hen_to_pigs:', hen_to_pigs, 'hen_to_eggs:', hen_to_eggs)
        # early finishing penalty
        early_finish = self._goal - egg_score if self._is_win(grid) else 0
        print('early_finish:', early_finish)
        # losing penalty
        lose_penalty = 1000 if not self.env.is_hen_exists(grid) or num_actions == self._actions_limit else 0
        print('lose penalty', lose_penalty)
        # calculate final utility
        utility = (
                3 * hen_safety +  # safety distance for the hen
                2 * bird_chase +  # chase efficiency for the bird
                1 * hen_to_endpoint +  # proximity to endpoint
                2 * hen_to_eggs -  # proximity to eggs
                0.1 * hen_to_pigs -  # proximity to eggs
                1 +  # price of action
                1 * early_finish -  # early finish penalty
                1 * lose_penalty  # lose penalty
        )
        print('utility:', utility)
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

    env = AngryGame(template='simple')

    screen, clock = PygameInit.initialization()
    FPS = 6

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
        best_action = minimax.get_best_actions(env.grid, env.num_actions, counter % 3, depth=DEPTH)

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
