import torch as T
from networks import ActorNetwork, CriticNetwork
from memory import Memory


class Agent:
    def __init__(self, actor_dims, critic_dims):
        self.actor_network = ActorNetwork(actor_dims, critic_dims)
        self.critic_network = CriticNetwork(actor_dims, critic_dims)
        self.target_actor_network = ActorNetwork(actor_dims, critic_dims)
        self.target_critic_network = CriticNetwork(actor_dims, critic_dims)
        self.memory = Memory()
        self.update_network_parameters(tau=1)       

    def actor(self, state):  # 根据状态state，给出一个确定的action
        state = T.tensor([state], dtype=T.double).to(self.actor_network.device)
        action = self.actor_network.forward(state)
        return action

    def target_actor(self, state):  # 根据状态state，给出一个确定的action
        state = T.tensor([state], dtype=T.double).to(self.actor_network.device)
        action = self.target_actor_network.forward(state)
        return action

    def critic(self, state, action):  # 在状态state下，给action打分，
        state = T.tensor([state], dtype=T.double).to(self.actor_network.device)
        val = self.critic_network.forward(state, action)
        return val

    def target_critic(self, state, action):  # 在状态state下，给action打分
        state = T.tensor([state], dtype=T.double).to(self.actor_network.device)
        val = self.target_critic_network.forward(state, action)
        return val

    # 注释修改了  更新actor网络 -> 更新target_actor网络
    def update_target_actor(self, TAU):  # 更新target_actor网络
        target_actor_params = self.target_actor_network.named_parameters()
        actor_params = self.actor_network.named_parameters()
        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = TAU*actor_state_dict[name].clone() + \
                    (1-TAU)*target_actor_state_dict[name].clone()
        self.target_actor_network.load_state_dict(actor_state_dict)

    def get_target_critic_parameter(self):  # 取得target_critic网络的参数
        target_critic_params = self.target_critic_network.named_parameters()
        target_critic_state_dict = dict(target_critic_params)
        return target_critic_state_dict

    def set_critic_parameter(self, theta):  # 设置critic网络的参数
        self.critic_network.load_state_dict(theta)

    def set_target_critic_parameter(self, theta_):  # 设置target_critic网络的参数
        self.target_critic_network.load_state_dict(theta_)

    def store(self, state, action, reward, state_):  # 把(s, a, r, s_)存入memory
        self.memory.save(state, action, reward, state_)

    def sample(self):  # 随机返回一个(s, a, r, s_)
        return self.memory.sample()
