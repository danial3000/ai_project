import pygame
from environment import UnknownAngryBirds, PygameInit

from qlearning import QLearning

if __name__ == "__main__":

    env = UnknownAngryBirds()
    screen, clock = PygameInit.initialization()
    FPS = 10

    ql = QLearning(env=env, decay_rate=.995, learning_rate=0.9, discount_factor=0.8, epsilon_greedy=0.99)
    values_difference, total_rewards = ql.explore(num_episodes=10000, conv_patience=10, conv_epsilon=10)
    ql.plot_values_difference(values_difference, total_rewards)
    policy = ql.set_policy()

    ql.plot_qtable_heatmap(ql.get_config_index([True for _ in range(8)]))
    ql.plot_policy(policy=policy)
    state = env.reset()

    episode_reward = []
    for _ in range(5):

        running = True
        total_reward = 0
        pig_state = [True for _ in range(8)]

        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            env.render(screen)

            action = policy[state[0], state[1], ql.get_config_index(pig_state)]
            next_state, reward, pig_state, done = env.step(action)
            state = next_state
            total_reward += reward

            if done:
                print(pig_state)
                print(f"Episode finished with reward: {total_reward}")
                state = env.reset()
                episode_reward.append(total_reward)
                total_reward = 0
                running = False

            pygame.display.flip()
            clock.tick(FPS)

    print(f'MEAN REWARD: {sum(episode_reward)/len(episode_reward)}')

    pygame.quit()