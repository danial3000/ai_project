import numpy as np
import pygame
from game import AngryGame, PygameInit


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

        if counter % 2 == 0:
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
