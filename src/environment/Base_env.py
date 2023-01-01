#TODO Base env -> DDQN environment 생성 후 정상 작동 확인 작업 (2023-01-01 ~)
from utils.utils import get_state,normalize

class env(object):
    def __init__(self,data):
        self.reward = None
        self.done = None
        self.history = []
        self.shares_history = []
        self.shares = 0
        self.net_holdings = 0
        self.total_profit = 0
        self.data = data
        self.normed_data = normalize(data)
        self.pct_change = self.daily_pct_change(data.price, 10)
        self.iter = 0
        pass

    def action(self,action):
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


    def step(self,action):
        self.action(action)
        self.shares_history.append(self.shares)
        self.reward = self.calc_reward(self.pct_change[self.iter] * 100, self.net_holdings)
        self.total_profit += self.reward

        if self.done == True:
            return self.total_profit, self.average_loss

        else:
            next_state = get_state(self.normed_data, self.iter + 1)
            return self.state,action,self.reward,next_state,self.done

            agent.remember(state, action, reward, next_state, done)
            state = next_state
        # step 
        pass

    def reset():
        # reset environment
        pass

    def render():
        pass

    def calc_reward(pct_change, net_holdings):
        return pct_change * net_holdings

    def daily_pct_change(prices, shift): #변화율 계산
        pct_change = (prices.copy() / prices.copy().shift(periods = shift)) - 1
        pct_change[:shift] = 0
        return pct_change
