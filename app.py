import datetime

import streamlit as st
import pandas as pd
import numpy as np

from src.BaselineModel import BaselineModel
from src.HeuristicTrader import HeuristicTrader
from src.methods import evaluate_model,evaluate_model_A2C,train_model_A2C,train_model
from src.agent import Agent,AC2_Agent
from src.utils import add_technical_features, load_data, results_df, plot_trades, get_portfolio_stats, plot_benchmark, \
  plot_benchmark2, load_data2,plot_benchmark_A2C_DDQN

@st.cache
def load_data_2(symbol, window_size):
  data_ = add_technical_features(load_data2(symbol), window=window_size).sort_values(by=['Date'], ascending=True)
  return data_

@st.cache
def load_data_(symbol, window_size):
  data_ = add_technical_features(load_data(f'data/{symbol}.csv'), window=window_size).sort_values(by=['Date'], ascending=True)
  return data_

@st.cache
def filter_data_by_date(data, start_date, end_date):
  """
  NA 결측치 데이터 삭제
  """
  return data[start_date:end_date].dropna()

def load_model(state_size, model_name):
  """
  학습 모델 Agent 호출
  """
  return Agent(state_size = window_size, pretrained = True, model_name = model_name)

def evaluate(agent, test_data, window_size, verbose = True):
  result, history, shares = evaluate_model(agent, test_data, window_size, verbose)
  return result, history, shares

def sidebar(index):
  start_date = st.sidebar.date_input('Start', index[0], min_value=index[0])
  end_date = st.sidebar.date_input('End', index[-1], max_value=index[-1])
  window_size = st.sidebar.slider('Window Size', 1, 30, 10)
  return start_date, end_date, window_size

def benchmarks(symbol, data, window_size = 10):
   baseline = BaselineModel(symbol, data, max_shares = 10)
   baseline_results = results_df(data.price, baseline.shares, starting_value = 1_000)
   heuristic = HeuristicTrader(symbol, data, window = window_size, max_shares = 10)
   heuristic_results = results_df(data.price, heuristic.shares, starting_value = 1_000)

   return baseline_results, heuristic_results

def get_heuristic_results(symbol, data, window_size=10):
  heuristic = HeuristicTrader(symbol, data, window=window_size, max_shares=10)
  heuristic_results = results_df(data.price, heuristic.shares, starting_value=1_000)

  return heuristic_results

# Streamlit App
st.title('DeepRL Trader')
st.header("DDQN 강화학습 모델 이용한 최적의 투자 전략 생성")
st.subheader('2022년 2학기 데이터분석캡스톤디자인 - F팀')

symbols = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM','JNJ', 'NFLX', 'SPY', 'TSLA']
symbol = st.sidebar.selectbox('Stock Symbol:', symbols)

index = load_data_(symbol, 10).index
start_date, end_date, window_size = sidebar(index)
submit = st.sidebar.button('Run')
submit2 = st.sidebar.button('Test')
submit3 = st.sidebar.button('jun-ac2 train')
submit4 = st.sidebar.button('jun-ac2 test')



if submit4:

  model_name = 'test'#symbol
  training_data = load_data_(symbol,window_size)
  filtered_data = filter_data_by_date(training_data, start_date, end_date)
  num_features = training_data.shape[1]
  
  """ A2C """
  def load_AC2_model(state_size, model_name):
      return AC2_Agent(state_size = num_features, pretrained = True, model_name = model_name)

  agent = load_AC2_model(filtered_data.shape[1], model_name = model_name)
  profit, history, shares = evaluate_model_A2C(agent, filtered_data, window_size = window_size, verbose = False)
  results = results_df(filtered_data.price, shares, starting_value = 1_000)
  cum_return, avg_daily_returns, std_daily_returns, sharpe_ratio = get_portfolio_stats(results.Port_Vals)

  st.write(f'### Cumulative Return for {symbol}: {np.around(cum_return * 100, 2)}%')
  fig = plot_trades(filtered_data, results.Shares, symbol)
  st.plotly_chart(fig)
  """  ================== """
  
  model_name = 'AAPL'
  dqn_agent = load_model(filtered_data.shape[1], model_name = model_name)
  dqn_profit, dqn_history, dqn_shares = evaluate(dqn_agent, filtered_data, window_size = window_size, verbose = False)
  dqn_results = results_df(filtered_data.price, dqn_shares, starting_value = 1_000)
  dqn_cum_return, dqn_avg_daily_returns, dqn_std_daily_returns, dqn_sharpe_ratio = get_portfolio_stats(dqn_results.Port_Vals)

  st.write(f'### Cumulative Return for {symbol}: {np.around(dqn_cum_return * 100, 2)}%')
  dqn_fig = plot_trades(filtered_data, dqn_results.Shares, symbol)
  st.plotly_chart(dqn_fig)



    ## Benchmarking
  ## BUY & hold, 휴리스틱 모델과 비교
  
  baseline_results, heuristic_results = benchmarks(symbol, filtered_data) 

  cum_return_base, avg_daily_returns_base, std_daily_returns_base, sharpe_ratio_base = get_portfolio_stats(baseline_results.Port_Vals)
  cum_return_heuristic, avg_daily_returns_heuristic, std_daily_returns_heuristic, sharpe_ratio_heuristic = get_portfolio_stats(heuristic_results.Port_Vals)

  benchmark = pd.DataFrame(columns = ['Cumulative Return', 'Avg Daily Returns', 'Std Dev Daily Returns', 'Sharpe Ratio'], index = ['A2C','Double DQN', 'Buy & Hold', 'Heuristic'])
  benchmark.loc['A2C'] = [cum_return * 100, avg_daily_returns * 100, std_daily_returns, sharpe_ratio]
  benchmark.loc['Double DQN'] = [dqn_cum_return * 100, dqn_avg_daily_returns * 100, dqn_std_daily_returns, dqn_sharpe_ratio]
  benchmark.loc['Heuristic' ] = [cum_return_heuristic * 100, avg_daily_returns_heuristic * 100, std_daily_returns_heuristic, sharpe_ratio_heuristic]
  #benchmark.loc['Buy & Hold'] = [cum_return_base * 100, avg_daily_returns_base * 100, std_daily_returns_base, sharpe_ratio_base]


  st.table(benchmark.astype('float64').round(4))

  fig = plot_benchmark_A2C_DDQN(heuristic_results, dqn_results,results,baseline_results)
  st.plotly_chart(fig)

  st.header('Raw Data')
  st.subheader('Double DQN')
  st.dataframe(results)

  #st.subheader('Buy & Hold')
  #st.write(baseline_results)

  st.subheader('Heuristic')
  st.write(heuristic_results)

if submit3: # jun ac2 train
  model_name = 'test'#symbol

  training_data = add_technical_features(load_data('data/GOOG.csv'), window = window_size).sort_values(by=['Date'], ascending=True)
  #training_data.to_csv("traiing.csv")
  validation_data = add_technical_features(load_data('data/GOOG_2018.csv'), window = window_size).sort_values(by=['Date'], ascending=True)
  num_features = training_data.shape[1]
  episode_count = 50; batch_size= 32; model_type = 'A2C'
  agent = AC2_Agent(state_size = num_features, model_type = model_type, model_name = model_name, window_size = window_size) 

  for episode in range(1, episode_count + 1):
    agent.n_iter += 1

    training_result = train_model_A2C(agent, episode, training_data, episode_count = episode_count, batch_size = batch_size, window_size = window_size)
    validation_profit, history, valid_shares = evaluate_model_A2C(agent, validation_data, verbose=True)

  
  #profit, history, shares = evaluate_model_A2C(agent, filtered_data, window_size = window_size, verbose = False)

  st.write('end')
  #profit, history, shares = evaluate(agent, filtered_data, window_size = window_size, verbose = False)

if submit2: #TEST
  model_name = symbol # 주식
  data = load_data_2(symbol, window_size)
  filtered_data = filter_data_by_date(data, start_date, end_date)

  agent = load_model(filtered_data.shape[1], model_name = model_name)
  profit, history, shares = evaluate(agent, filtered_data, window_size = window_size, verbose = False)
  results = results_df(filtered_data.price, shares, starting_value = 1_000)
  cum_return, avg_daily_returns, std_daily_returns, sharpe_ratio = get_portfolio_stats(results.Port_Vals)

  st.write(f'### Cumulative Return for {symbol}: {np.around(cum_return * 100, 2)}%')
  fig = plot_trades(filtered_data, results.Shares, symbol)
  st.plotly_chart(fig)

  ## Benchmarking
  ## BUY & hold, 휴리스틱 모델과 비교
  
  baseline_results, heuristic_results = benchmarks(symbol, filtered_data) 

  cum_return_base, avg_daily_returns_base, std_daily_returns_base, sharpe_ratio_base = get_portfolio_stats(baseline_results.Port_Vals)
  cum_return_heuristic, avg_daily_returns_heuristic, std_daily_returns_heuristic, sharpe_ratio_heuristic = get_portfolio_stats(heuristic_results.Port_Vals)

  benchmark = pd.DataFrame(columns = ['Cumulative Return', 'Avg Daily Returns', 'Std Dev Daily Returns', 'Sharpe Ratio'], index = ['Double DQN', 'Buy & Hold', 'Heuristic'])
  benchmark.loc['Double DQN'] = [cum_return * 100, avg_daily_returns * 100, std_daily_returns, sharpe_ratio]
  benchmark.loc['Heuristic' ] = [cum_return_heuristic * 100, avg_daily_returns_heuristic * 100, std_daily_returns_heuristic, sharpe_ratio_heuristic]
  benchmark.loc['Buy & Hold'] = [cum_return_base * 100, avg_daily_returns_base * 100, std_daily_returns_base, sharpe_ratio_base]


  st.table(benchmark.astype('float64').round(4))

  fig = plot_benchmark(baseline_results, heuristic_results, results)
  st.plotly_chart(fig)

  st.header('Raw Data')
  st.subheader('Double DQN')
  st.dataframe(results)

  st.subheader('Buy & Hold')
  st.write(baseline_results)

  st.subheader('Heuristic')
  st.write(heuristic_results)


  ## Benchmarking2

  heuristic_results = get_heuristic_results(symbol, filtered_data)
  cum_return_heuristic, avg_daily_returns_heuristic, std_daily_returns_heuristic, sharpe_ratio_heuristic = get_portfolio_stats(heuristic_results.Port_Vals)

  benchmark = pd.DataFrame(columns = ['Cumulative Return', 'Avg Daily Returns', 'Std Dev Daily Returns', 'Sharpe Ratio'], index = ['Double DQN', 'Buy & Hold', 'Heuristic'])
  benchmark.loc['Double DQN'] = [cum_return * 100, avg_daily_returns * 100, std_daily_returns, sharpe_ratio]
  benchmark.loc['Heuristic' ] = [cum_return_heuristic * 100, avg_daily_returns_heuristic * 100, std_daily_returns_heuristic, sharpe_ratio_heuristic]


  st.table(benchmark.astype('float64').round(4))

  fig = plot_benchmark2(heuristic_results, results)
  st.plotly_chart(fig)

  st.header('Raw Data')
  st.subheader('Double DQN')
  st.dataframe(results)

  st.subheader('Heuristic')
  st.write(heuristic_results)

if submit: #TEST
  model_name = symbol
  data = load_data_(symbol, window_size)
  filtered_data = filter_data_by_date(data, start_date, end_date)

  agent = load_model(filtered_data.shape[1], model_name = model_name)
  profit, history, shares = evaluate(agent, filtered_data, window_size = window_size, verbose = False)
  results = results_df(filtered_data.price, shares, starting_value = 1_000)
  cum_return, avg_daily_returns, std_daily_returns, sharpe_ratio = get_portfolio_stats(results.Port_Vals)

  st.write(f'### Cumulative Return for {symbol}: {np.around(cum_return * 100, 2)}%')
  fig = plot_trades(filtered_data, results.Shares, symbol)
  st.plotly_chart(fig)