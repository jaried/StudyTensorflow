import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

env = gym.make('MountainCar-v0')
env.seed(0)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])


# class for drawing
class Chart:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)
        # plt.ion()

    def plot(self, episode_rewards):
        self.ax.clear()
        self.ax.plot(episode_rewards)
        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('episode reward')
        self.fig.canvas.draw()


class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward',
                                            'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


class DQNAgent:
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.001,
                 replayer_capacity=10000, batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)  # Experience playback

        self.evaluate_net = self.build_network(input_size=observation_dim, output_size=self.action_n,
                                               **net_kwargs)  # evaluate the network
        self.target_net = self.build_network(input_size=observation_dim, output_size=self.action_n,
                                             **net_kwargs)  # target network

        self.target_net.set_weights(self.evaluate_net.get_weights())

    @staticmethod
    def build_network(input_size, hidden_sizes, output_size,
                      activation=tf.nn.relu, output_activation=None,
                      learning_rate=0.01):  # build network
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation))  # output layer
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation, done)  # store experience

        # Experience playback
        observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)
        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - dones) * next_max_qs
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:  # update target network
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def decide(self, observation):  # epsilon greedy strategy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluate_net.predict(observation[np.newaxis])
        return np.argmax(qs)


def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation, done)
        if done:
            break
        observation = next_observation
    return episode_reward


render = False
# Deep Q-Network Solving Optimal Policy
net_kwargs = {'hidden_sizes': [64, 64], 'learning_rate': 0.001}
agent = DQNAgent(env, net_kwargs=net_kwargs)

# traning
episodes = 500
episode_rewards = []
chart = Chart()
for episode in tqdm(range(episodes), total=episodes):
    episode_reward = play_qlearning(env, agent, train=True, render=render)
    episode_rewards.append(episode_reward)
    chart.plot(episode_rewards)

# testing
agent.epsilon = 0.  # Cancel exploration
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print(
    'Average round reward = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))
