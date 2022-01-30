class Agent:
    def __init__(self):
        # 成员变量如下：
        #    一个memory
        #    一个critic网络
        #    一个actor网络
        pass

    def actor(self, state):  # 根据状态state，给出一个确定的action
        pass

    def target_actor(self, state):  # 根据状态state，给出一个确定的action
        pass

    def critic(self, state, action):  # 在状态state下，给action打分，
        pass

    def target_critic(self, state, action):  # 在状态state下，给action打分
        pass

    def update_actor(self, loss):  # 更新actor网络
        pass

    def get_target_critic_parameter(self):  # 取得target_critic网络的参数
        pass

    def set_target_critic_parameter(self):  # 设置target_critic网络的参数
        pass

    def store(self, state, action, reward, state_):  # 把(s, a, r, s_)存入memory
        pass

    def sample(self):  # 随机返回一个(s, a, r, s_)
        pass
