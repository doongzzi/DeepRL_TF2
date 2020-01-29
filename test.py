import sys
import gym
import pylab
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layer
import tensorflow.keras.optimizers as opt
import tensorflow.keras as keras

x = tf.constant([[1., 1.], [2., 2.]])
print(tf.reduce_mean(x))  # 1.5
print(tf.reduce_mean(x, 0))  # [1.5, 1.5]
print(tf.reduce_mean(x, 1))  # [1.,  2.]

print(tf.__version__)
