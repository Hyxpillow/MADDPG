from agent import Agent
from server import Server
import numpy as np
import gym

# 计算loss需要用
########################
import torch
import torch.nn.functional as F
########################*/


# 在执行MADDPG算法之前进行相关变量的初始化


env = gym.make('Pendulum-v0')
M = 1000
MAX_EPISODE_LENGTH = 2000
N = 1
GAMMA = 0.9
TAU = 0.01
STATE_DIMENSION = env.observation_space.shape[0]
ACTION_DIMENSION = env.action_space.shape[0]

state_list = [0 for i in range(N)]
agent_list = [Agent(STATE_DIMENSION, ACTION_DIMENSION) for i in range(N)]
server = Server(N, STATE_DIMENSION, ACTION_DIMENSION)





a_bound = env.action_space.high
a_low_bound = env.action_space.low
var = 3
# 真正开始执行MADDPG
cnt = 3
for episode in range(M):
    torch.autograd.Variable
    state = env.reset()
    state_list = [state] * N
    env.render()
    
    for step in range(MAX_EPISODE_LENGTH):
        
        for i, agent_i in enumerate(agent_list):
            action = agent_i.actor(state_list[i])
            action = np.clip(np.random.normal(action, var), a_low_bound, a_bound)
            state_, reward, done, info = env.step(action)
            agent_i.store(state_list[i], action, reward, state_)
            state_list[i] = state_
        var *= 0.9995 # decay the exploration controller factor
        for i, agent_i in enumerate(agent_list):
            state, action, reward, state_ = agent_i.sample()
            
            action_ = agent_i.target_actor(state_)
            y_ = reward + GAMMA * agent_i.target_critic(state_, action_)
            y = agent_i.critic(state, action)
            
            #server.recv(i, y, y_)
            #server.update_critic()




            y = torch.tensor(y, requires_grad=True, dtype=torch.float)
            y_ = torch.tensor(y_, requires_grad=True, dtype=torch.float)
            loss = F.mse_loss(y_, y)
            server.critic_network.optimizer.zero_grad()
            loss.backward()
            optimizer = torch.optim.Adam(server.critic_network.parameters(), lr=0.001)
            #server.critic_network.optimizer.step()
            optimizer.step()
            #theta = server.send()
            #if step  3:
                #print(theta)


            loss = agent_i.critic(state, action)
            agent_i.update_actor(loss)

            
            theta = server.send()
            if step > 3:
                print(theta)
            

        theta = server.send()

        for agent_i in agent_list:
            if cnt > 0:
                print(theta)
            cnt -= 1
            agent_i.set_critic_parameter(theta)
            agent_i.update_target_actor(TAU)
            agent_i.update_target_critic(TAU)

        if done:
            break
env.close()
