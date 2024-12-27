import numpy as np
import pygame
from game import AngryGame, PygameInit


class MiniMax:
    _goal = 1600
    _actions_limit = 200

    def __init__(self, environment):
        self.env = environment
        self.num_actions = environment.num_actions
        self.endpoint = environment.get_slingshot_position(environment.grid)

    @staticmethod
    def distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def heuristic_function(self, grid, num_actions):
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
        early_finish = self._goal - total_score if self.env.is_lose(grid, num_actions) or self.env.is_win(grid) else 0

        # losing penalty
        lose_penalty = 1000 if not self.env.is_hen_exists(grid) or num_actions == self._actions_limit else 0

        # calculate final utility
        utility = (
                2 * total_score +  # prioritize collected points
                30 * hen_safety +  # safety distance for the hen
                30 * bird_chase +  # chase efficiency for the bird
                30 * hen_to_endpoint +  # proximity to endpoint
                10 * hen_to_eggs -  # proximity to eggs
                2 * hen_to_pigs -  # proximity to eggs
                1 * num_actions +  # number of actions
                1 * hen_score +  # score of eggs eaten by the hen
                1 * bird_score -  # score of the bird eating the queen
                1 * early_finish -  # early finish penalty
                1 * lose_penalty  # lose penalty
        )

        return utility

if __name__ == "__main__":

    env = AngryGame(template='simple')

    screen, clock = PygameInit.initialization()
    FPS = 2

    env.reset()
    counter = 0

    running = True
    while running:
        if AngryGame.is_win(env.grid) or AngryGame.is_lose(env.grid, env.num_actions):
           running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        if counter % 3 == 0:
            action = np.random.choice([0, 1, 2, 3])
            env.hen_step(action)
            env.render(screen)
            if AngryGame.is_win(env.grid):
                running = False

        if counter % 3 == 1:
            action = np.random.choice([0, 1, 2, 3])
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
