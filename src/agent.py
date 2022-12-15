from collections import deque
import random

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.losses import Huber
from keras.models import clone_model
from keras.optimizers import Adam
#from keras.saving.save import load_model tf 2.0.0
from keras.models import load_model # tf 2.11.0

from src.utils import timestamp
import math

class Agent:
  def __init__(self, state_size, model_type = 'ddqn', pretrained = False, model_name = None, window_size = 10, reset_target_weight_interval = 10):
    self.model_type = model_type

    self.state_size = state_size
    self.action_size = 3
    self.inventory = []
    self.memory = deque(maxlen = 10000)
    self.start = True

    self.model_name = model_name
    self.gamma = 0.99
    self.rar = 0.99 # Epsilon / Random Action Rate
    self.eps_min = 0.01
    self.radr = 0.995 # Random Action Decay Rate
    self.lr = 1e-5
    self.loss = Huber
    self.custom_objects = {"huber": Huber}
    self.optimizer = Adam(lr = self.lr)
    self.window_size = window_size

    if pretrained and self.model_name is not None:
      self.model = self.load()
    else:
      self.model = self.model_()

    self.n_iter = 1
    self.reset_interval = reset_target_weight_interval

    self.target_model = clone_model(self.model)
    self.target_model.set_weights(self.model.get_weights())


  def load(self):
    model = load_model(f"models/{self.model_name}", custom_objects=self.custom_objects, compile=False)
    model.compile(optimizer=self.optimizer, loss=self.loss())
    return model

  def save(self, episode):
    if self.model_name is None:
      self.model_name = f'{self.model_type}_{timestamp()}'
    print(f'================={self.model.get_weights()}=====================')
    self.model.save(f"models/{self.model_name}_{episode}")

  def model_(self):
    model = Sequential()
    model.add(Dense(units=256, activation="relu", input_shape=(self.state_size,)))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=self.action_size))

    model.compile(optimizer = self.optimizer, loss = self.loss())
    return model

  def action(self, state, evaluation = False):
    if self.start:
      self.start = False
      return 1

    if not evaluation and (random.random() <= self.rar):
      return random.randrange(self.action_size)

    action_probs = self.model.predict(state)
    print(action_probs)
    return np.argmax(action_probs[0])

  def replay(self, batch_size):
    mini_batch = random.sample(self.memory, batch_size)
    X_train, y_train = [], []

    if self.model_type == 'ddqn':
      if self.n_iter % self.reset_interval == 0:
        print("Setting Target Weights...")
        self.target_model.set_weights(self.model.get_weights())

      for state, action, reward, next_state, done in mini_batch:
        if done:
          target = reward
        else:
          target = reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]

        q_values = self.model.predict(state)
        q_values[0][action] = target
        X_train.append(state[0])
        y_train.append(q_values[0])

    if self.rar > self.eps_min:
      self.rar *= self.radr

    loss = self.model.fit(
      x  = np.array(X_train),
      y = np.array(y_train),
      epochs = 1,
      verbose = 0
    ).history["loss"][0]

    return loss

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


###########################################################################


#import wandb
import tensorflow as tf
from keras.layers import Input, Dense
from keras import Model
from keras.models import save_model,load_model

#import gym
import argparse
import numpy as np
import random

args = {'actor_lr' : 0.0005, 'gamma' : 0.99, 'update_interval' : 5, 'critic_lr' : 0.001}
'''
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)

args = parser.parse_args()
'''

## keras 와 tensorflow.python.keras 섞어서 사용하면 안됨.

class Actor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.opt = Adam(lr=args['actor_lr'])
        self.lr = 0.001
        self.loss = Huber
        self.model = self.create_model()
        print('=================',self.opt)

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=32, activation="relu", input_shape=(self.state_dim,)))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=self.action_dim,activation='softmax'))
        model.compile(optimizer=self.opt, loss=self.loss())
        return model

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        return policy_loss

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(
                actions, logits, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.opt = tf.keras.optimizers.Adam(args['critic_lr'])
        self.loss = Huber
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(Dense(units=32, activation="relu", input_shape=(self.state_dim,)))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=1,activation='linear'))
        model.compile(optimizer=self.opt, loss=self.loss())
        #model.compile(optimizer=tf.keras.optimizers.Adam(lr =0.001),loss=Huber)
        return model

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class AC2_Agent:
    def __init__(self, state_size, model_type = 'ddqn', pretrained = False, model_name = None, window_size = 10, reset_target_weight_interval = 10):

        self.model_type = model_type
        self.state_size = state_size
        self.action_size = 3
        self.inventory = []
        self.start = True
        
        self.model_name = model_name
        self.actor = Actor(self.state_size, self.action_size)
        self.critic = Critic(self.state_size)
        self.custom_objects = {"huber": Huber}
        self.rar = 0.99 # Epsilon / Random Action Rate
        self.window_size = window_size
        self.loss = Huber
        
        if pretrained and self.model_name is not None:
            print("model loaded -================")
            self.actor.model,self.critic.model = self.load()
            #print(f'loaded weihgt : {self.actor_model.get_weights()}')
            print("========================================")
          

        self.n_iter = 1
        self.reset_interval = reset_target_weight_interval

        self.actor_target_model = clone_model(self.actor.model)
        self.actor_target_model.set_weights(self.actor.model.get_weights())
        self.critic_target_model = clone_model(self.critic.model)
        self.critic_target_model.set_weights(self.critic.model.get_weights())
        

    def td_target(self, reward, next_state, done):
        if done:
            return reward
        v_value = self.critic.model.predict(next_state)
            #np.reshape(next_state, [1, self.state_size]))
        return np.reshape(reward + args['gamma'] * v_value[0], [1, 1])

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def load(self):
        actor_opt_weights = np.load('models/A2C_actor_opt_weights.npy',allow_pickle=True)
        critic_opt_weights = np.load('models/A2C_critic_opt_weights.npy',allow_pickle=True)
        actor_model = load_model(f"models/A2C_actor_{self.model_name}.h5")#, custom_objects=self.custom_objects, compile=False)#, custom_objects=self.custom_objects, compile=False)
        critic_model = load_model(f"models/A2C_critic_{self.model_name}.h5")#,  custom_objects=self.custom_objects, compile=False)
        #print(f'loaded before weihgt : {actor_model.get_weights()}')

        actor_grad_vars = actor_model.trainable_weights
        critic_grad_vars = critic_model.trainable_weights

        actor_zero_grads = [tf.zeros_like(w) for w in actor_grad_vars]
        critic_zero_grads = [tf.zeros_like(w) for w in critic_grad_vars]

        self.actor.opt.apply_gradients(zip(actor_zero_grads, actor_grad_vars))
        self.critic.opt.apply_gradients(zip(critic_zero_grads, critic_grad_vars))

        self.actor.opt.set_weights(actor_opt_weights)
        self.critic.opt.set_weights(critic_opt_weights)
        
        actor_model.compile(optimizer=self.actor.opt, loss=self.loss())
        critic_model.compile(optimizer=self.critic.opt, loss=self.loss())
        #print(f'loading {actor_model.get_weights()}')
        return actor_model,critic_model

    def save(self, actor,critic,episode):
        if self.model_name is None:
            self.model_name = f'{self.model_type}_{timestamp()}'
        print(f'save================={actor.get_weights()}=============')
        print(f'save================={critic.get_weights()}===============')
        actor.save_weights(f"models/A2C_actor_{self.model_name}.h5")
        critic.save_weights(f"models/A2C_critic_{self.model_name}.h5")

    def action(self, state, evaluation = False):
        if self.start:
            self.start = False
            return 1

        probs = self.actor.model.predict(state)
        #print(f' state is : {state}')
        print(f' what is : {probs[0]}')
        #print(f'type is :{np.isnan(probs[0][0])}')
        if np.isnan(probs[0][0]) == True:
            #print(random.randrange(self.action_size))
            return  random.randrange(self.action_size)
            
        else:
          #print(f'================ {probs[0]} ================')
          _action = np.random.choice(self.action_size, p=probs[0])
          print(probs,"============== arg max : ",_action)
          return _action
        





