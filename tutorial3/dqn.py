import tensorflow as tf
import numpy as np

np.random.seed(2)
tf.set_random_seed(2)


class DeepQNetwork:
    def __init__(self, n_actions, n_features, learning_rate=0.5, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=1000, batch_size=32, e_greedy_increment=None, output_graph=False,
                 use_pre_weights=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment else self.epsilon_max

        self.learn_step_counter = 0
        # [s, a, r, s_] = 2*features + 2
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        if output_graph:
            tf.summary.FileWriter('./net_model', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("./saved_net_params")
        if use_pre_weights and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)

    def _build_net(self):
        # input
        self.s = tf.placeholder(tf.float32, [None, self.n_features], 's')
        self.a = tf.placeholder(tf.int32, [None, ], 'a')
        self.r = tf.placeholder(tf.float32, [None, ], 'r')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], 's_')

        w_initializer = tf.truncated_normal_initializer(0., 0.5)
        b_initializer = tf.constant_initializer(0.1)

        n_hidden_units = 10
        # eval_net
        with tf.variable_scope('eval_net'):
            # with tf.variable_scope('e_fcl1'):
            #     w1 = tf.get_variable('w1', [self.n_features, n_hidden_units], initializer=w_initializer)
            #     b1 = tf.get_variable('b1', [1, n_hidden_units], initializer=b_initializer)
            #     e_l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            #
            # with tf.variable_scope('e_fcl2'):
            #     w2 = tf.get_variable('w2', [n_hidden_units, n_hidden_units], initializer=w_initializer)
            #     b2 = tf.get_variable('b2', [1, n_hidden_units], initializer=b_initializer)
            #     e_l2 = tf.matmul(e_l1, w2) + b2
            e_l1 = tf.layers.dense(self.s, n_hidden_units, tf.nn.relu, kernel_initializer=w_initializer,
                                   bias_initializer=b_initializer, name='e_fcl1')
            e_l2 = tf.layers.dense(e_l1, n_hidden_units, tf.nn.relu, kernel_initializer=w_initializer,
                                   bias_initializer=b_initializer, name='e_fcl2')
            self.q_eval = tf.layers.dense(e_l2, self.n_actions, tf.nn.relu, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_evel')

        # target net
        with tf.variable_scope('target_net'):
            t_l1 = tf.layers.dense(self.s_, n_hidden_units, tf.nn.relu, kernel_initializer=w_initializer,
                                   bias_initializer=b_initializer, name='t_fcl1')
            t_l2 = tf.layers.dense(t_l1, n_hidden_units, tf.nn.relu, kernel_initializer=w_initializer,
                                   bias_initializer=b_initializer, name='t_fcl2')
            self.q_next = tf.layers.dense(t_l2, self.n_actions, tf.nn.relu, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_next')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='q_max_s_')
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_a = tf.gather_nd(self.q_eval, a_indices)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_a, name='TemporalDiff_error'))
        with tf.name_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_val = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_val, axis=1)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('target params are replaced at step {0}'.format(self.learn_step_counter))

        if self.memory_counter > self.memory_size:
            sample_idx = np.random.choice(self.memory_size, self.batch_size)
        else:
            sample_idx = np.random.choice(self.memory_counter, self.batch_size)

        sample_batch = self.memory[sample_idx, :]

        _, loss = self.sess.run([self._train_op, self.loss],
                                feed_dict={
                                    self.s: sample_batch[:, :self.n_features],
                                    self.a: sample_batch[:, self.n_features],
                                    self.r: sample_batch[:, self.n_features + 1],
                                    self.s_: sample_batch[:, -self.n_features:],
                                })

        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increment

        self.learn_step_counter += 1


if __name__ == '__main__':
    DQN = DeepQNetwork(3, 4, output_graph=True)
