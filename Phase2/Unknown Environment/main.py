import pygame
from environment import UnknownAngryBirds, PygameInit

from QLearning import QLearning

if __name__ == "__main__":

    env = UnknownAngryBirds()
    screen, clock = PygameInit.initialization()
    FPS = 10

    ql = QLearning(env)
    ql.explore(1000)
    policy = ql.setPolicy()
    state = env.reset()

    episode_reward = []
    for _ in range(5):

        running = True
        total_reward = 0
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            env.render(screen)

            action = policy[state]
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode finished with reward: {total_reward}")
                state = env.reset()
                episode_reward.append(total_reward)
                total_reward = 0
                running = False

            pygame.display.flip()
            clock.tick(FPS)

    print(f'MEAN REWARD: {sum(episode_reward)/len(episode_reward)}')

    pygame.quit()