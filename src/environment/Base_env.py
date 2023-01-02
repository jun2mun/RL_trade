#TODO Base env -> DDQN environment 생성 후 정상 작동 확인 작업 (2023-01-01 ~)
from src.utils.utils import get_state,normalize

class env(object):
    def __init__(self,data):
        self.reward = None
        self.done = None

        self.shares = 0
        self.net_holdings = 0
        self.total_profit = 0
        self.data = data
        self.pct_change = self.daily_pct_change(self.data.price, 10)
        self.current_step = 0

        self.state = [] # history, shares, 
        pass

    def act(self,action):
        if action == 2 and self.net_holdings == 0:
            self.shares = -10
            self.net_holdings += -10
            self.history = (self.data.price[self.current_step], "SELL")
        elif action == 2 and self.net_holdings == 10:
            self.shares = -20
            self.net_holdings += -20
            self.history = (self.data.price[self.current_step], "SELL")
        elif action == 1 and self.net_holdings == 0:
            self.shares = 10
            self.net_holdings += 10
            self.history = (self.data.price[self.current_step], "BUY")
        elif action == 1 and self.net_holdings == -10:
            self.shares = 20
            self.net_holdings += 20
            self.history = (self.data.price[self.current_step], "BUY")
        else:
            self.shares = 0
            self.history = (self.data.price[self.current_step], "HOLD")


    def step(self,action):
        self.done = self.current_step == (len(self.data) -1)
        
        self.act(action)
        self.state = [self.history,self.shares]
        self.reward = self.calc_reward(self.pct_change[self.current_step] * 100, self.net_holdings)
        self.total_profit += self.reward
        self.current_step += 1
        return self.state,self.reward,self.done

            

    def reset(self):
        # reset environment
        self.net_holdings = 0
        self.net_holdings = 0

    def render():
        pass

    def calc_reward(self,pct_change, net_holdings):
        return pct_change * net_holdings

    def daily_pct_change(self,prices, shift): #변화율 계산
        pct_change = (prices.copy() / prices.copy().shift(periods = shift)) - 1
        pct_change[:shift] = 0
        return pct_change
