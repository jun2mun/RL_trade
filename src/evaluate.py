import argparse
import logging
import os

import coloredlogs

from agent import Agent,AC2_Agent
from methods import evaluate_model,evaluate_model_A2C
from utils import add_technical_features, load_data, show_evaluation_result
import keras.backend as K


def run(eval_stock, window_size, model_name, model_type ,verbose):
  data = add_technical_features(load_data(eval_stock), window = window_size).sort_values(by=['Date'], ascending=True)
  num_features = data.shape[1]

  if model_name is not None:
    if model_type == 'ddqn':
      agent = Agent(num_features, pretrained=True, model_name=model_name)
      profit, history, valid_shares = evaluate_model(agent, data, verbose)
      show_evaluation_result(profit)
    elif model_type == 'A2C':
      agent = AC2_Agent(num_features, pretrained=True, model_name=model_name)
      profit, history, valid_shares = evaluate_model_A2C(agent, data, verbose)
      show_evaluation_result(profit)


  #if model_name is not None:
  #  agent = Agent(num_features, pretrained=True, model_name=model_name)
  #  profit, history, valid_shares = evaluate_model(agent, data, verbose)
  #  show_evaluation_result(profit)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate RLAgent')
  parser.add_argument('--eval')
  parser.add_argument('--window-size', default = 10)
  parser.add_argument('--model-name')
  parser.add_argument('--model-type',default='ddqn')
  parser.add_argument('--verbose', default = True)

  args = parser.parse_args()

  eval_stock = args.eval
  window_size = int(args.window_size)
  model_name = args.model_name
  verbose = args.verbose
  model_type = args.model_type

  coloredlogs.install(level="DEBUG")

  if K.backend() == "tensorflow":
    logging.debug("Switching --> TensorFlow for CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  try:
    run(eval_stock, window_size, model_name, model_type,verbose)
  except KeyboardInterrupt:
    print("Aborted with Keyboard Interrupt..")