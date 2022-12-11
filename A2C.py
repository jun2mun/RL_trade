#import wandb
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import save_model,load_model

#import gym
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)

args = parser.parse_args()

from src.utils import timestamp


class Actor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(Dense(units=32, activation="relu", input_shape=(self.state_dim,)))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=self.action_dim,activation='softmax'))
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
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(Dense(units=32, activation="relu", input_shape=(self.state_dim,)))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=1,activation='linear'))
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
    def __init__(self,state_size,pretrained=False,model_name=None):
        #elf.model_type = model_type
        #self.model_name ='test'
        self.model_name = model_name
        self.state_size = state_size
        self.action_size = 3
        self.start = True
        self.inventory = []
        self.actor = Actor(self.state_size, self.action_size)
        self.critic = Critic(self.state_size)

        self.rar = 0.99 # Epsilon / Random Action Rate

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.actor = Actor(self.state_size, self.action_size)
            self.critic = Critic(self.state_size)
        
    def td_target(self, reward, next_state, done):
        if done:
            return reward
        v_value = self.critic.model.predict(next_state)
            #np.reshape(next_state, [1, self.state_size]))
        return np.reshape(reward + args.gamma * v_value[0], [1, 1])

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def load(self):
        model = load_model(f"models/{self.model_name}")#, custom_objects=self.custom_objects, compile=False)
        #model.compile(optimizer=self.actor.opt, loss=self.actor.train())
        return model

    def save(self, episode):
        if self.model_name is None:
            self.model_name = f'{self.model_type}_{timestamp()}'
            self.model.save(f"models/A2C_{self.model_name}_{episode}")

    def action(self, state, evaluation = False):
        if self.start:
            self.start = False
            return 1

        if not evaluation and (random.random() <= self.rar):
            return random.randrange(self.action_size)

        action_probs = self.actor.model.predict(state)
        return np.argmax(action_probs[0])
