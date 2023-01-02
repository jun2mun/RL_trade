from src.environment.Base_env import env
class DDQN_env(env):
    def __init__(self,data):
        super().__init__(data)
        