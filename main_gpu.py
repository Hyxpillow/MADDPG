# -*- coding: utf-8 -*-
import copy
from xmlrpc.client import Server
from make_env import make_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
np.random.seed(1)

# 类定义

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(ActorNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.max_action = max_action
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.to(self.device)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

    def choose_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        return self.forward(s).detach().cpu()  # single action


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, x, u):
        temp_ = torch.cat((x, u), 1)
        temp = self.l1(temp_)
        x = F.relu(temp)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        a = Act522(a)
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


class Agent:
    def __init__(self, state_dim, action_dim, num_of_agents):
        self.target_actor = ActorNetwork(state_dim, action_dim)
        self.actor = ActorNetwork(state_dim, action_dim)
        self.num_of_agents = num_of_agents
        self.state_dim = state_dim
        self.action_dim = action_dim


class Server:
    def __init__(self, state_dim, action_dim, num_of_agents, memory_list):
        self.critic_list = [CriticNetwork(state_dim, action_dim) for i in range(num_of_agents)]
        self.target_critic_list = [CriticNetwork(state_dim, action_dim) for i in range(num_of_agents)]
        self.memory_list = memory_list
        self.num_of_agents = num_of_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

# 函数定义
def Act225(a):
    a = [0, a[0], 0, a[1], 0]
    return [a]


def Act522(a):
    a = [a[1], a[3]]
    return a

# 参数定义
MAX_EPISODES = 25000
MAX_EP_STEPS = 25
LR_A = 1e-3  # learning rate for actor
LR_C = 1e-3  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 100000
BATCH_SIZE = 1024
VAR_MIN = 0
TAU = 0.01
RENDER = True
LOAD = False
MODE = ['easy', 'hard']
n_model = 1

game = 'simple'
env = make_env(game)
STATE_DIM = 4
ACTION_BOUND = [-1, 1]

STATE_PUSH_ADVERSARY = 8
STATE_PUSH_GOOD = 19

STATE_DIM_RED = 8
STATE_DIM_BLUE = 10

STATE_DIM_SPREAD = 18

ACTION_DIM = 2
NUM_OF_AGENTS = len(env.agents)
RENDER_AFTER_EP = 4200

# 初始化
if game == 'simple_adversary':
    agents_list = [Agent(STATE_DIM_RED, ACTION_DIM, NUM_OF_AGENTS),
                Agent(STATE_DIM_BLUE, ACTION_DIM, NUM_OF_AGENTS),
                Agent(STATE_DIM_BLUE, ACTION_DIM, NUM_OF_AGENTS)]
elif game == 'simple':
    agents_list = [Agent(STATE_DIM, ACTION_DIM, NUM_OF_AGENTS)]
elif game == 'simple_push':
    agents_list = [Agent(STATE_PUSH_ADVERSARY, ACTION_DIM, NUM_OF_AGENTS),
                Agent(STATE_PUSH_GOOD, ACTION_DIM, NUM_OF_AGENTS)]
elif game == 'simple_spread':
    agents_list = [Agent(STATE_DIM_SPREAD, ACTION_DIM, NUM_OF_AGENTS),
                Agent(STATE_DIM_SPREAD, ACTION_DIM, NUM_OF_AGENTS),
                Agent(STATE_DIM_SPREAD, ACTION_DIM, NUM_OF_AGENTS)]

server_state_dim = 0
server_action_dim = 0
for agent in agents_list:
    server_state_dim += agent.state_dim
    server_action_dim += agent.action_dim

memory_list = [Memory(capacity=MEMORY_CAPACITY, dims=2 * agent.state_dim + agent.action_dim + 1) for agent in agents_list]
server = Server(server_state_dim, server_action_dim, NUM_OF_AGENTS, memory_list)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def sample(memory_list, agents_list):
    states = np.zeros((BATCH_SIZE, 0))
    actions = np.zeros((BATCH_SIZE, 0))
    rewards = np.zeros((BATCH_SIZE, 0))
    states_ = np.zeros((BATCH_SIZE, 0))
    for i, agent in enumerate(agents_list):
        M = memory_list[i]
        
        b_M = M.sample(BATCH_SIZE)
        b_s = b_M[:, :agent.state_dim]
        b_a = b_M[:, agent.state_dim: agent.state_dim + agent.action_dim]
        b_r = b_M[:, -agent.state_dim - 1: -agent.state_dim]
        b_s_ = b_M[:, -agent.state_dim:]

        states = np.hstack((states, b_s))
        actions = np.hstack((actions, b_a))
        rewards = np.hstack((rewards, b_r))
        states_ = np.hstack((states_, b_s_))
    # print("states_", states_)
    
    states = torch.FloatTensor(states).to(device)  # 转换成tensor类型
    actions = torch.FloatTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    states_ = torch.FloatTensor(states_).to(device)
    return states, actions, rewards, states_

def pre_process(agents_list):
    idx_list = [[0, 0] for i in range(len(agents_list))]
    idx_left = 0
    for i, agent in enumerate(agents_list):
        idx_list[i][0] += idx_left
        idx_list[i][1] = idx_list[i][0] + agent.state_dim
        idx_left = idx_list[i][1]
    return idx_list

def train():
    print(device)
    idx_list = pre_process(agents_list)
    print(idx_list)
    var = 2.  # control exploration
    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        
        for t in range(MAX_EP_STEPS):
            var = max([var * .9999, VAR_MIN])  # decay the action randomness
            if ep > RENDER_AFTER_EP:
                env.render()
                time.sleep(0.01)
            actions = []
            for i, agent in enumerate(agents_list):
                a = agent.actor.choose_action(s[i])
                a = np.clip(np.random.normal(a, var), *ACTION_BOUND)  # add randomness to action selection for exploration
                a = Act225(a)
                actions += a
            s_, r, done, _ = env.step(actions)

            for i, agent in enumerate(agents_list):
                M = server.memory_list[i]
                M.store_transition(s[i], actions[i], r[i], s_[i])
            s = s_
            ep_reward += sum(r)
            if server.memory_list[0].pointer <= MEMORY_CAPACITY:
                continue

            # 用于存放全局tensor
            for i, agent in enumerate(agents_list):
                #start = time.time()
                states, actions, rewards, states_ = sample(server.memory_list, agents_list)
                #print("sample_time:", time.time() - start)
                

                server.critic_list[i].optimizer.zero_grad()
                
                #start = time.time()
                actions_ = torch.FloatTensor([]).to(device)
                for i_, agent_ in enumerate(agents_list):
                    a_ = agent_.target_actor.forward(states_[:, idx_list[i_][0]:idx_list[i_][1]])
                    actions_ = torch.cat((actions_, a_), 1)
                #print("target_actions_time:", time.time() - start)
                # 更新critic

                #start = time.time()
                y_ = rewards[:, i:i+1] + GAMMA * server.target_critic_list[i].forward(states_, actions_)
                y = server.critic_list[i].forward(states, actions)
                td_error = F.mse_loss(y_.detach(), y)
                
                # if ep > RENDER_AFTER_EP:
                #     print("td_error", td_error.data)
                
                td_error.backward()
                torch.nn.utils.clip_grad_norm_(server.critic_list[i].parameters(), 0.5)
                server.critic_list[i].optimizer.step()
                #print("update_critic_time:", time.time() - start)

                agent.actor.optimizer.zero_grad()

                #start = time.time()
                _actions = torch.FloatTensor([]).to(device)
                temp = 0
                for i_, agent_ in enumerate(agents_list):
                    if i_ == i:
                        temp = agent_.actor.forward(states[:, idx_list[i_][0]:idx_list[i_][1]])
                        a = temp
                    else: 
                        a = actions[:, i_ * agent.action_dim : (i_ + 1) * agent.action_dim]
                    _actions = torch.cat((_actions, a), 1)
                #print("old_action_time:", time.time() - start)
                # 本地更新actor
                #start = time.time()
                loss = server.critic_list[i].forward(states, _actions)
                    
                # for parameters in server.critic_list[i].l1.parameters():
                #     print(parameters)
                actor_loss = -torch.mean(loss)
                actor_loss += (temp ** 2).mean() * 1e-3
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                agent.actor.optimizer.step()
                #print("update_actor_time:", time.time() - start)


            for i, agent in enumerate(agents_list):
                #start = time.time()
                for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
                for target_param, param in zip(server.target_critic_list[i].parameters(), server.critic_list[i].parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
                #print("softupdate_time:", time.time() - start)
            
            

            if ep % 10 == 0 and (t == MAX_EP_STEPS - 1 or done[0]):
                print('=================')
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      )
                break


if __name__ == '__main__':
    train()