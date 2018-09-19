import numpy as np
import pandas as pd
import time
import pdb

N_STATE = 10
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 15
FRESH_TIME = 0.05

np.random.seed(1)


def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)

    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()

    return action_name


def get_env_feedback(state, action):
    if action == 'right':
        if state == N_STATE - 2:
            state_ = 'terminal'
            reward = 1
        else:
            state_ = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            state_ = state
        else:
            state_ = state - 1

    return state_, reward


def update_env(state, episode, step_counter):
    env_list = ['-'] * (N_STATE - 1) + ['$']
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(1)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def reinforcement_learning():
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, step_counter)
        while not is_terminated:
            action = choose_action(state, q_table)
            state_, reward = get_env_feedback(state, action)
            q_predict = q_table.loc[state, action]
            if state_ != 'terminal':
                q_target = reward + GAMMA * q_table.iloc[state_, :].max()
            else:
                q_target = reward
                is_terminated = True

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            state = state_

            update_env(state, episode, step_counter + 1)
            step_counter += 1
        pdb.set_trace()
    return q_table


if __name__ == '__main__':
    Q_table = reinforcement_learning()
    print('\r\nQ-table:\n')
    print(Q_table)
