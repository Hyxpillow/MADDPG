import torch as T
from networks import ActorNetwork, CriticNetwork
##每个Agent都有一个buffer
from memory import Memory

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, gamma=0.95, tau=0.01):
        # 成员变量如下：
        #    一个memory
        #    一个critic网络
        #    一个actor网络
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor_network = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.critic_network = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor_network = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic_network = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')
        # 相关输入参数需要重新制定
        self.memory = Memory(1000000, critic_dims, actor_dims, 
                        n_actions, batch_size=1024)
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
