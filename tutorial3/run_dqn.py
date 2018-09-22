from maze_env import Maze
from dqn import DeepQNetwork


def train():
    step = 0
    episode = 0
    while True:
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        episode += 1
        if episode % 100 == 0:
            RL.saver.save(RL.sess, 'saved_net_params/' + 'maze-dqn', global_step=episode)
            print('Save params at episode {0}'.format(episode))
            break

    print('Game Over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      use_pre_weights=True)
    env.after(1000, train())
    env.mainloop()
