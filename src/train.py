import argparse
import logging
import os

import coloredlogs as coloredlogs
import keras.backend as K

from agent import Agent,AC2_Agent

from methods import train_model, evaluate_model, train_model_A2C, evaluate_model_A2C
from utils import load_data, add_technical_features, show_training_result


def run(training_stock, validation_stock, window_size, batch_size, episode_count, model_type="ddqn", model_name = None, pretrained = False, verbose = False):
  training_data = add_technical_features(load_data(training_stock), window = window_size).sort_values(by=['Date'], ascending=True)
  validation_data = add_technical_features(load_data(validation_stock), window = window_size).sort_values(by=['Date'], ascending=True)

  num_features = training_data.shape[1]
  if model_type == "ddqn":
    agent = Agent(state_size = num_features, model_type = model_type, model_name = model_name, window_size = window_size)

    for episode in range(1, episode_count + 1):
      agent.n_iter += 1

      training_result = train_model(agent, episode, training_data, episode_count = episode_count, batch_size = batch_size, window_size = window_size)
      validation_profit, history, valid_shares = evaluate_model(agent, validation_data, verbose)

      show_training_result(training_result, validation_profit)

  elif model_type == "A2C":
    agent = AC2_Agent(state_size = num_features, model_type = model_type, model_name = model_name, window_size = window_size) 

    for episode in range(1, episode_count + 1):
      agent.n_iter += 1

      training_result = train_model_A2C(agent, episode, training_data, episode_count = episode_count, batch_size = batch_size, window_size = window_size)
      validation_profit, history, valid_shares = evaluate_model_A2C(agent, validation_data, verbose)

      show_training_result(training_result, validation_profit)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Deep RL in Algo Trading')
  parser.add_argument('--train')
  parser.add_argument('--valid')
  parser.add_argument('--model-type', default = 'ddqn')
  parser.add_argument('--window-size', default = 10)
  parser.add_argument('--batch-size', default = 32)
  parser.add_argument('--episode-count', default = 50)
  parser.add_argument('--model-name')
  parser.add_argument('--pretrained', default = False)
  parser.add_argument('--verbose', default = False)

  args = parser.parse_args()

  training_stock = args.train
  validation_stock = args.valid
  model_type = args.model_type
  window_size = int(args.window_size)
  batch_size = int(args.batch_size)
  episode_count = int(args.episode_count)


  model_name = args.model_name
  pretrained = args.pretrained
  verbose = args.verbose

  coloredlogs.install(level = "DEBUG")

  if K.backend() == "tensorflow":
    logging.debug("Switching --> TensorFlow for CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
  print("hello")
  try:
    run(training_stock, validation_stock, window_size, batch_size, episode_count,
    model_type=model_type, pretrained = pretrained, verbose = verbose,model_name=model_name)
  except KeyboardInterrupt:
    print("Aborted with Keyboard Interrupt..")
