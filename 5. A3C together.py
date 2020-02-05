import sys
import gym
import pylab
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layer
import tensorflow.keras.optimizers as opt
import tensorflow.keras as keras
import threading
import time

env_global = gym.make('CartPole-v1')
state_size = env_global.observation_space.shape[0]
action_size = env_global.action_space.n


class coordinate:

    def __init__(self):

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.state_size = state_size
        self.action_size = action_size
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.threads = 8

    def train(self):

        agents = [A3C_Agent(i, state_size, action_size, self.actor, self.critic)
                  for i in range(self.threads)]

        for agent in agents:
            time.sleep(1)
            agent.start()

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


class A3C_Agent(threading.Thread):
    def __init__(self, name, state_size, action_size, actor, critic):
        threading.Thread.__init__(self)
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        #self.batch_size = 32
        self.state_size = state_size
        self.action_size = action_size
        self.actor = actor
        self.critic = critic
        self.local_actor = self.build_actor()
        self.local_critic = self.build_critic()
        self.agent_name = name

    def build_actor(self):
        actor = keras.models.Sequential()
        actor.add(layer.Dense(24, input_dim=self.state_size, activation='relu'))
        actor.add(layer.Dense(24, input_dim=self.state_size, activation='relu'))
        actor.add(layer.Dense(self.action_size, activation='softmax'))
        actor.summary()
        actor.set_weights(self.actor.get_weights())
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
        critic.set_weights(self.critic.get_weights())
        critic.summary()
        critic.compile(loss='mse', optimizer=opt.Adam(lr=0.005))
        return critic

    def network_update(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def select_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def run(self):
        env = gym.make('CartPole-v1')
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
                reward = reward if not done or score == 499 else -100
                next_state = np.reshape(next_state, [1, state_size])
                advantage = np.zeros((1, self.action_size))
                target = np.zeros((1, 1))

                V_t = self.local_critic.predict(state)[0]
                V_t1 = self.local_critic.predict(next_state)[0]
                advantage[0][action] = reward + \
                    self.discount_factor * V_t1*(1-done) - V_t
                target[0][0] = advantage[0][action] + V_t

                self.actor.fit(state, advantage, epochs=1, verbose=0)
                self.critic.fit(state, target, epochs=1, verbose=0)
                self.network_update()

                score += reward
                state = next_state

                if done:
                    score = score if score == 500 else score+100
                    print("Agent #: "+str(self.agent_name) +
                          ", episode:"+str(e)+", score="+str(score))
                    state = env.reset()
                    rewards.append(score)
                    score = 0


if __name__ == "__main__":
    global_coord = coordinate()
    global_coord.train()
