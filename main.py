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
env = gym.make('Pendulum-v1')
M = 1000
MAX_EPISODE_LENGTH = 200
N = 1
GAMMA = 0.05
TAU = 0.05
STATE_DIMENSION = env.observation_space.shape[0]
ACTION_DIMENSION = env.action_space.shape[0]

state_list = [0 for i in range(N)]
agent_list = [Agent(STATE_DIMENSION, ACTION_DIMENSION) for i in range(N)]
server = Server(N, STATE_DIMENSION, ACTION_DIMENSION)


# 真正开始执行MADDPG
for episode in range(M):
    state = env.reset()
    state_list = [torch.tensor(state, dtype=torch.float)] * N
    for step in range(MAX_EPISODE_LENGTH):
        # env.render()
        for i, agent_i in enumerate(agent_list):
            action = agent_i.actor(state_list[i])
            state_, reward, done, info = env.step(action.detach().numpy())
            state_ = torch.tensor(state_, dtype=torch.float)
            reward = torch.tensor(reward, dtype=torch.float)
            agent_i.store(state_list[i], action, reward, state_)
            state_list[i] = state_

        for i, agent_i in enumerate(agent_list):
            agent_i.actor_network.optimizer.zero_grad()

        for i, agent_i in enumerate(agent_list):
            state, action, reward, state_ = agent_i.sample()
            action_ = agent_i.target_actor(state_)
            y_ = reward + GAMMA * agent_i.target_critic(state_, action_)
            y = agent_i.critic(state, action)
            server.recv(i, (y, y_))

            actor_loss = agent_i.critic(state, action)
            actor_loss = -torch.mean(actor_loss)
            actor_loss.backward(retain_graph=True)

        for i, agent_i in enumerate(agent_list):
            agent_i.actor_network.optimizer.step()
# 目前黄禹翔debug到这里，有事出门去了A.A
###########################################################
###########################################################
        server.update_critic()
        theta = server.send()


# 增加更新agent的critic网络和target_actor网络
############################################################
        # 可能出现的问题，梯度未清零，只是猜测，以便以后debug
        for agent_i in agent_list:
            agent_i.set_critic_parameter(theta)
            agent_i.update_target_actor(TAU)
############################################################
        theta_ = agent_list[0].get_target_critic_parameter()
        # 待修改，因为存在名字对应问题
        theta_ = TAU * theta + (1 - TAU) * theta_
        for agent_i in agent_list:
            agent_i.set_target_critic_parameter(theta_)

        # if done: # 如果游戏结束，则跳出step循环，进入下一episode；否则，继续step
        #     break
env.close()
