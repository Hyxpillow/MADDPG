from networks import CriticNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F

class Server:
    def __init__(self, num_of_agents, actor_dims, critic_dims):
        # 拥有的成员变量包括:
        self.critic_network = CriticNetwork(actor_dims, critic_dims)
        self.loss_buffer = [0] * num_of_agents
        #  一个用来存放所有Loss值的buffer

    def recv(self, agent_index, y, y_):  # 从agent那收到一个loss
        self.loss_buffer[agent_index] = (y, y_)

    def send(self):
        critic_params = self.critic_network.named_parameters()
        critic_state_dict = dict(critic_params)
        return critic_state_dict

    def update_critic(self):
        loss = F.mse_loss(self.loss_buffer[0][0], self.loss_buffer[0][1])
        self.critic_network.optimizer.zero_grad()
        loss.backward()
        self.critic_network.optimizer.step()



