from maze_env import Maze
from rl_brain import QLearningTable, SarsaTable, SarsaLambdaTable

METHOD = ['q_learning', 'sarsa', 'sarsa_lambda']


def update(method):
    if method == METHOD[0]:
        for episode in range(100):
            observation = env.reset()
            while True:
                env.render()
                action = RL.choose_action(str(observation))
                observation_, reward, done = env.step(action)
                RL.learn(str(observation), action, reward, str(observation_))
                observation = observation_
                if done:
                    break;

    elif method == METHOD[1]:
        for episode in range(100):
            observation = env.reset()
            action = RL.choose_action(str(observation))
            while True:
                env.render()
                observation_, reward, done = env.step(action)
                action_ = RL.choose_action(str(observation_))
                RL.learn(str(observation), action, reward, str(observation_), action_)
                action = action_
                observation = observation_
                if done:
                    break;

    elif method == METHOD[2]:
        for episode in range(100):
            observation = env.reset()
            action = RL.choose_action(str(observation))
            RL.eligibility_trace *= 0
            while True:
                env.render()
                observation_, reward, done = env.step(action)
                action_ = RL.choose_action(str(observation_))
                RL.learn(str(observation), action, reward, str(observation_), action_)
                action = action_
                observation = observation_
                if done:
                    break;

    print('Game Over')
    env.destroy()


if __name__ == '__main__':
    method = 'sarsa_lambda'
    env = Maze()
    if method == METHOD[0]:
        RL = QLearningTable(actions=list(range(env.n_actions)))
        env.after(1000, update(METHOD[0]))
    elif method == METHOD[1]:
        RL = SarsaTable(actions=list(range(env.n_actions)))
        env.after(1000, update(METHOD[1]))
    elif method == METHOD[2]:
        RL = SarsaLambdaTable(actions=list(range(env.n_actions)))
        env.after(1000, update(METHOD[2]))
    env.mainloop()
