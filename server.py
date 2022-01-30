from networks import CriticNetwork


class Server:
    def __init__(self, num_of_agents, actor_dims, critic_dims):
        # 拥有的成员变量包括:
        self.critic_network = CriticNetwork(actor_dims, critic_dims)
        self.loss_buffer = [0] * num_of_agents
        #  一个用来存放所有Loss值的buffer

    def recv(self, agent_index, loss):  # 从agent那收到一个loss
        self.loss_buffer[agent_index] = loss

    def send(self):
        pass

    def update_critic(self):
        pass

