# DQN에서 action 선택하는 부분과 target 계산하는 부분만 바꿔서 완성
import sys
import gym
import pylab
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layer
import tensorflow.keras.optimizers as opt


class DQN_Agent:
    def __init__(self, state_size, action_size):

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_max = 3000
        self.replay_buffer = []

        self.model = self.build_model()
        self.target_model = self.build_model()

    def store_buffer(self, state, action, reward, next_state, done):
        if(len(self.replay_buffer) >= self.buffer_max):
            del self.replay_buffer[0]
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_buffer(self):
        return random.sample(self.replay_buffer, self.batch_size)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, input_dim=state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        model.summary()
        model.compile(optimizer=opt.Adam(
            learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_model(self):
        mini_batch = self.sample_buffer()
        states = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        self.epsilon = max(self.epsilon_decay*self.epsilon, self.epsilon_min)
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_state[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model.predict(states)
        target_v = self.model.predict(next_state)
        target_q = self.target_model.predict(next_state)
        for i in range(self.batch_size):
            target[i][actions[i]] = rewards[i] + self.discount_factor*(
                # np.amax(target_q[i])
                target_q[i][np.argmax(target_v[i])]
            )*(1-dones[i])
        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN_Agent(state_size, action_size)
    scores = []
    for e in range(2000):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100
            agent.store_buffer(state, action, reward, next_state, done)
            if(len(agent.replay_buffer) >= agent.train_start):
                agent.train_model()

            score += reward
            state = next_state

            if done:
                agent.update_target()
                score = score if score == 500 else score+100
                scores.append(score)
                print("episode:", e, "score: ", score, "memory  len :",
                      len(agent.replay_buffer), "epsilon: ", agent.epsilon)
            if np.mean(scores[-min(10, len(scores)):]) > 490:
                sys.exit()
