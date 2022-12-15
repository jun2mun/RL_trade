import os
import logging
import numpy as np
from tqdm import tqdm
from src.utils import get_state, format_currency, format_position, normalize
import pdb
import streamlit as st
import tensorflow as tf
'''
1. Move daily_pct_return to utils
2. Move calc_reward to utils
'''

def advatnage(td_targets, baselines):
    return td_targets - baselines

def list_to_batch(list):
    batch = list[0]
    for elem in list[1:]:
        batch = np.append(batch, elem, axis=0)
    return batch

def daily_pct_change(prices, shift): #변화율 계산
  pct_change = (prices.copy() / prices.copy().shift(periods = shift)) - 1
  pct_change[:shift] = 0
  return pct_change

def calc_reward(pct_change, net_holdings):
  return pct_change * net_holdings

def train_model(agent, episode, data, episode_count = 50, batch_size = 32, window_size = 10):
  total_profit = 0
  num_observations = len(data)

  agent.inventory = []
  shares_history = []
  average_loss = []

  net_holdings = 0
  normed_data = normalize(data)
  pct_change = daily_pct_change(data.price, window_size)

  for t in tqdm(range(num_observations), total = num_observations, leave = True, desc = f'Episode {episode}/{episode_count}'):
    done = t == (num_observations - 1)

    state = get_state(normed_data, t)
    action = agent.action(state)

    if action == 2 and net_holdings == 0:
        shares = -100
        net_holdings += -100
    elif action == 2 and net_holdings == 100:
        shares = -200
        net_holdings += -200
    elif action == 1 and net_holdings == 0:
        shares = 100
        net_holdings += 100
    elif action == 1 and net_holdings == -100:
        shares = 200
        net_holdings += 200
    else:
        shares = 0
    shares_history.append(shares)

    reward = calc_reward(pct_change[t] * 100, net_holdings)
    total_profit += reward

    # if action == 1: # Buy
    #   agent.inventory.append(data.price[t])
    #
    #   reward -= 1e-5 # Commission Penalty

    # elif action == 2 and len(agent.inventory) > 0: # Sell
    #   purchase_price = agent.inventory.pop(0)
    #   delta = data.price[t] - purchase_price
    #   reward = delta - 1e-5 # Commission Penalty
    #   total_profit += delta
    #   shares.append(-1)

    # else: # Hold
    #   shares.append(0)
    #   reward -= 1e-3

    if not done:
      next_state = get_state(normed_data, t + 1)
      agent.remember(state, action, reward, next_state, done)
      state = next_state

    if len(agent.memory) > batch_size:
      loss = agent.replay(batch_size)
      average_loss.append(loss)

    if episode % 2 == 0:
      agent.save(episode)

    if done: return (episode, episode_count, total_profit, np.array(average_loss).mean())

def evaluate_model(agent, data, verbose, window_size = 10):
  total_profit = 0
  num_observations = len(data)

  shares = []
  history = []
  agent.inventory = []
  normed_data = normalize(data)
  cum_return = []
  net_holdings = 0
  shares_history = []
  pct_change = daily_pct_change(data.price, 10)

  for t in range(num_observations):
    done = t == (num_observations - 1)
    reward = 0


    state = get_state(normed_data, t)
    action = agent.action(state, evaluation = True)

    if action == 2 and net_holdings == 0:
      shares = -10
      net_holdings += -10
      history.append((data.price[t], "SELL"))
    elif action == 2 and net_holdings == 10:
      shares = -20
      net_holdings += -20
      history.append((data.price[t], "SELL"))
    elif action == 1 and net_holdings == 0:
      shares = 10
      net_holdings += 10
      history.append((data.price[t], "BUY"))
    elif action == 1 and net_holdings == -10:
      shares = 20
      net_holdings += 20
      history.append((data.price[t], "BUY"))
    else:
      shares = 0
      history.append((data.price[t], "HOLD"))
    shares_history.append(shares)

    reward = calc_reward(pct_change[t], net_holdings)
    total_profit += reward
    # if action == 1:
    #   agent.inventory.append(data.price[t])
    #   shares.append(1)
    #   history.append((data.price[t], "BUY"))

    #   if verbose:
    #     logging.debug(f"Buy at: {format_currency(data.price[t])}")

    # elif action == 2 and len(agent.inventory) > 0:
    #   purchase_price = agent.inventory.pop(0)
    #   delta = data.price[t] - purchase_price
    #   reward = delta
    #   total_profit += delta
    #   shares.append(-1)
    #   history.append((data.price[t], "SELL"))

    #   if verbose:
    #     logging.debug(f"Sell at: {format_currency(data.price[t])} | Position: {format_position(data.price[t] - purchase_price)}")

    # else:
    #   history.append((data.price[t], "HOLD"))
    #   shares.append(0)
    # cum_return.append(total_profit)

    if not done:
      next_state = get_state(normed_data, t + 1)
      agent.memory.append((state, action, reward, next_state, done))
      state = next_state

    if done: return total_profit, history, shares_history

#####  수정중
def train_model_A2C(agent, episode, data, episode_count = 50, batch_size = 32, window_size = 10):
    total_profit = 0
    num_observations = len(data)

    agent.inventory = []
    shares_history = []
    average_loss = []

    state_batch = []
    action_batch = []
    td_target_batch = []
    advatnage_batch = []

    net_holdings = 0
    normed_data = normalize(data)
    pct_change = daily_pct_change(data.price, window_size)
    for t in tqdm(range(num_observations), total = num_observations, leave = True, desc = f'Episode {episode}/{episode_count}'):
      done = t == (num_observations - 1)

      state = get_state(normed_data, t)
      action = agent.action(state)
      if action == 2 and net_holdings == 0:
          shares = -100
          net_holdings += -100
      elif action == 2 and net_holdings == 100:
          shares = -200
          net_holdings += -200
      elif action == 1 and net_holdings == 0:
          shares = 100
          net_holdings += 100
      elif action == 1 and net_holdings == -100:
          shares = 200
          net_holdings += 200
      else:
          shares = 0
      shares_history.append(shares)

      reward = calc_reward(pct_change[t] * 100, net_holdings)
      total_profit += reward

      if not done:
        next_state = get_state(normed_data, t + 1)
        #st.write(next_state.shape) # (1,33)

        action = np.reshape(action, [1, 1])
        reward = np.reshape(reward, [1, 1])


        td_target = agent.td_target(reward = reward * 0.01, next_state=next_state, done=done)
        advantage = advatnage(
          td_target, agent.critic.model.predict(state))


        state_batch.append(state)
        action_batch.append(action)
        td_target_batch.append(td_target)
        advatnage_batch.append(advantage)

        state = next_state

      if t % batch_size == 0:
          loss = td_target
          average_loss.append(loss)

      if done:
          states = list_to_batch(state_batch)
          actions = list_to_batch(action_batch)
          td_targets = list_to_batch(td_target_batch)
          advantages = list_to_batch(advatnage_batch)


          with tf.GradientTape() as tape:
              logits = agent.actor_target_model(states, training=True)
              loss = agent.actor.compute_loss(
                  actions, logits, advantages)
          grads = tape.gradient(loss, agent.actor_target_model.trainable_variables)
          agent.actor.opt.apply_gradients(zip(grads, agent.actor_target_model.trainable_variables))
          actor_loss = loss

          with tf.GradientTape() as tape:
              v_pred = agent.critic_target_model(states, training=True)
              assert v_pred.shape == td_targets.shape
              loss = agent.critic.compute_loss(v_pred, tf.stop_gradient(td_targets))
          grads = tape.gradient(loss, agent.critic_target_model.trainable_variables)
          agent.critic.opt.apply_gradients(zip(grads, agent.critic_target_model.trainable_variables))
          critic_loss = loss
          
          
          if episode % 2 == 0:
              np.save("models/A2C_actor_opt_weights.npy",agent.actor.opt.get_weights())
              np.save("models/A2C_critic_opt_weights.npy",agent.critic.opt.get_weights())
              agent.save(episode,agent.actor_target_model,agent.critic_target_model)          

          state_batch = []
          action_batch = []
          td_target_batch = []
          advatnage_batch = []

          #return (episode, episode_count, total_profit)

          return (episode, episode_count, total_profit, np.array(actor_loss+critic_loss).mean())




def evaluate_model_A2C(agent, data, verbose, window_size = 10):
  total_profit = 0
  num_observations = len(data)

  shares = []
  history = []
  agent.inventory = []
  normed_data = normalize(data)
  cum_return = []
  net_holdings = 0
  shares_history = []
  pct_change = daily_pct_change(data.price, 10)

  for t in range(num_observations):
    done = t == (num_observations - 1)
    reward = 0


    state = get_state(normed_data, t)
    action = agent.action(state, evaluation = True)
    print(f' eval : {agent.actor_target_model.get_weights()}')

    if action == 2 and net_holdings == 0:
      shares = -10
      net_holdings += -10
      history.append((data.price[t], "SELL"))
    elif action == 2 and net_holdings == 10:
      shares = -20
      net_holdings += -20
      history.append((data.price[t], "SELL"))
    elif action == 1 and net_holdings == 0:
      shares = 10
      net_holdings += 10
      history.append((data.price[t], "BUY"))
    elif action == 1 and net_holdings == -10:
      shares = 20
      net_holdings += 20
      history.append((data.price[t], "BUY"))
    else:
      shares = 0
      history.append((data.price[t], "HOLD"))
    shares_history.append(shares)

    reward = calc_reward(pct_change[t], net_holdings)
    total_profit += reward

    if not done:
        next_state = get_state(normed_data, t + 1)
        #st.write(next_state.shape) # (1,33)

        state = next_state

    if done: return total_profit, history, shares_history


