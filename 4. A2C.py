# DDQN에서 Network부분만 떼내어 duel로 구성하여 완성. tf reduce를 빼주고 레이어합하는 부분을 잘 봐둘것.
# https://jsideas.net/dqn/ 매우 도움되는 사이트. 참고.
import sys
import gym
import pylab
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layer
import tensorflow.keras.optimizers as opt
import tensorflow.keras as keras


class A2C_Agent:
    def __init__(self, state_size, action_size):

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        #self.batch_size = 32
        self.state_size = state_size
        self.action_size = action_size

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        actor = keras.models.Sequential()
        actor.add(layer.Dense(24, input_dim=self.state_size, activation='relu'))
        actor.add(layer.Dense(24, input_dim=self.state_size, activation='relu'))
        actor.add(layer.Dense(self.action_size, activation='softmax'))
        actor.summary()
        actor.compile(loss='categorical_crossentropy',
                      optimzer=opt.Adam(lr=0.001))
        return actor

    def build_critic(self):
        critic = keras.models.Sequential()
        critic.add(layer.Dense(
            24, input_dim=self.state_size, activation='relu'))
        critic.add(layer.Dense(
            24, input_dim=self.state_size, activation='relu'))

        critic.add(layer.Dense(1, activation='linear'))
        critic.summary()
        critic.compile(loss='mse', optimizer=opt.Adam(lr=0.005))
        return critic

    def select_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def train_model(self):

        for e in range(2000):
            #states = np.zeros((self.batch_size, self.state_size))
            #next_state = np.zeros((self.batch_size, self.state_size))
            actions, rewards, dones = [], [], []
            state = env.reset()
            state = np.reshape(state, [1, state_size])

            done = False
            score = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                advantage = np.zeros((1, self.action_size))
                target = np.zeros((1, 1))

                V_t = self.critic.predict(state)[0]
                V_t1 = self.critic.predict(next_state)[0]
                advantage[0][action] = reward + \
                    self.discount_factor * V_t1*(1-done) - V_t
                target[0][0] = advantage[0][action] + V_t

                self.actor.fit(state, advantage, epochs=1, verbose=0)
                self.critic.fit(state, target, epochs=1, verbose=0)

                score += reward
                state = next_state

                if done:
                    print("episode:"+str(e)+", score="+str(score))
                    state = env.reset()
                    rewards.append(score)
                    score = 0


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = A2C_Agent(state_size, action_size)
    agent.train_model()
