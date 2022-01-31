from networks import CriticNetwork
import torch

class Server:
    def __init__(self, num_of_agents, actor_dims, critic_dims):
        # 拥有的成员变量包括:
        self.critic_network = CriticNetwork(actor_dims, critic_dims)
        self.loss_buffer = [0] * num_of_agents
        #  一个用来存放所有Loss值的buffer

    def recv(self, agent_index, loss):  # 从agent那收到一个loss
        self.loss_buffer[agent_index] = loss

    def send(self):
        critic_params = self.critic_network.named_parameters()
        critic_state_dict = dict(critic_params)
        return critic_state_dict

    def update_critic(self):
        self.critic_network.optimizer.zero_grad()
        self.loss_buffer[0].backward(retain_graph=True)
        self.critic_network.optimizer.step()



