import torch
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork
from memory import Memory


class Agent:
    def __init__(self, actor_dims, critic_dims):
        self.actor_network = ActorNetwork(actor_dims, critic_dims)
        self.critic_network = CriticNetwork(actor_dims, critic_dims)
        self.target_actor_network = ActorNetwork(actor_dims, critic_dims)
        self.target_critic_network = CriticNetwork(actor_dims, critic_dims)
        self.memory = Memory()

    def actor(self, state):  # 根据状态state，给出一个确定的action
        action = self.actor_network.forward(state)
        return action

    def target_actor(self, state):  # 根据状态state，给出一个确定的action
        action = self.target_actor_network.forward(state)
        return action

    def critic(self, state, action):  # 在状态state下，给action打分，
        val = self.critic_network.forward(state, action)
        return val

    def target_critic(self, state, action):  # 在状态state下，给action打分
        val = self.target_critic_network.forward(state, action)
        return val

    def update_actor(self, loss):
        actor_loss = -torch.mean(loss)
        self.actor_network.optimizer.zero_grad()
        actor_loss.backward()
        self.actor_network.optimizer.step()

    def update_critic(self, y, y_):  # 实际不会用到这个方法，写出来是为了测试
        td_error = F.mse_loss(y_, y)
        self.critic_network.optimizer.zero_grad()
        td_error.backward()
        self.critic_network.optimizer.step()

        critic_params = self.critic_network.named_parameters()
        critic_state_dict = dict(critic_params)
        return critic_state_dict

    def update_target_actor(self, TAU):  # 更新target_actor网络
        target_actor_params = self.target_actor_network.named_parameters()
        actor_params = self.actor_network.named_parameters()
        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = TAU*actor_state_dict[name].clone() + \
                    (1-TAU)*target_actor_state_dict[name].clone()
        self.target_actor_network.load_state_dict(actor_state_dict)

    def update_target_critic(self, TAU):  # 更新target_actor网络
        target_critic_params = self.target_critic_network.named_parameters()
        critic_params = self.critic_network.named_parameters()
        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = TAU*critic_state_dict[name].clone() + \
                    (1-TAU)*target_critic_state_dict[name].clone()
        self.target_critic_network.load_state_dict(critic_state_dict)

    def get_critic_parameter(self):  # 获取critic网络的参数
        return self.critic_network.named_parameters()

    def set_critic_parameter(self, theta):  # 设置critic网络的参数
        self.critic_network.load_state_dict(theta)

    def store(self, state, action, reward, state_):  # 把(s, a, r, s_)存入memory
        self.memory.save(state, action, reward, state_)

    def sample(self):  # 随机返回一个(s, a, r, s_)
        return self.memory.sample()
