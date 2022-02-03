from networks import CriticNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F

class Server:
    def __init__(self, num_of_agents, actor_dims, critic_dims):
        # 拥有的成员变量包括:
        self.critic_network = CriticNetwork(actor_dims, critic_dims)
        self.model_buffer = [0] * num_of_agents #  一个用来存放所有模型参数的buffer
        

     # 从agent那收到网络模型参数
    def recv(self, agent_index, y): 
        self.model_buffer[agent_index] = y

    # 返回模型平均值
    def send(self):
        return self.average()
    
    # 设置critic网络的参数
    def set_critic_parameter(self, theta):  
        self.critic_network.load_state_dict(theta)
    
    # 更新critic网络
    def update_critic(self, theta):  
        self.set_critic_parameter(theta)
    
    # 取agent模型参数的平均值
    def average(self):
        theta = [0]
        for i in self.model_buffer:
            theta += i
        theta /= self.num_of_agents
        return theta
