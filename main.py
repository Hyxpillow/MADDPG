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
    state_list = [state] * N
    for step in range(MAX_EPISODE_LENGTH):
        env.render()
        for i, agent_i in enumerate(agent_list):
            action = agent_i.actor(state_list[i])
            state_, reward, done, info = env.step(action)
            agent_i.store(state_list[i], action, reward, state_)
            state_list[i] = state_

        for i, agent_i in enumerate(agent_list):
            state, action, reward, state_ = agent_i.sample()
            action_ = agent_i.target_actor(state_)
            y_ = reward + GAMMA * agent_i.target_critic(state_, action_)
            y = agent_i.critic(state, action)
            server.recv(i, y, y_)
            server.update_critic()

            loss = agent_i.critic(state, action)
            agent_i.update_actor(loss)

        theta = server.send()

        for agent_i in agent_list:
            agent_i.set_critic_parameter(theta)
            agent_i.update_target_actor(TAU)
            agent_i.update_target_critic(TAU)

        if done:
            break
env.close()
